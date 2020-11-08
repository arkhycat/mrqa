import math
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from eval import eval_qa
from iterator import read_squad_examples, convert_examples_to_features
from model import DomainQA
from utils import eta, progress_bar, summary_map

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def get_opt(param_optimizer, num_train_optimization_steps, args):
    """
    Hack to remove pooler, which is not used
    Thus it produce None grad that break apex
    """
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return BertAdam(optimizer_grouped_parameters,
                    lr=args.lr,
                    warmup=args.warmup_proportion,
                    t_total=num_train_optimization_steps)


def make_weights_for_balanced_classes(classes, n_classes):
    count = [0] * n_classes
    for c in classes:
        count[c] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(classes)
    for idx, val in enumerate(classes):
        weight[idx] = weight_per_class[val]
    return weight


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.set_random_seed(random_seed=args.random_seed)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                                       do_lower_case=args.do_lower_case)
        if args.debug:
            print("Debugging mode on.")
        self.features_lst, self.num_to_name = self.get_features(self.args.train_folder, self.args.debug)

    def make_model_env(self, gpu, ngpus_per_node):
        if self.args.distributed:
            self.args.gpu = self.args.devices[gpu]
        else:
            self.args.gpu = 0

        if self.args.use_cuda and self.args.distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        # Load baseline model
        self.model = BertForQuestionAnswering.from_pretrained(self.args.bert_model)

        if self.args.load_model is not None:
            print("Loading model from ", self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model, map_location=lambda storage, loc: storage))

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) * self.args.epochs * len(self.features_lst)

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        self.optimizer = get_opt(list(self.model.named_parameters()), num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model.cuda()
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def make_run_env(self):
        if self.args.distributed:
            # distributing dev file evaluation task
            self.dev_files = []
            gpu_num = len(self.args.devices)
            files = os.listdir(self.args.dev_folder)
            for i in range(len(files)):
                if i % gpu_num == self.args.rank:
                    self.dev_files.append(files[i])

            print("GPU {}".format(self.args.gpu), self.dev_files)
        else:
            self.dev_files = os.listdir(self.args.dev_folder)
            print(self.dev_files)

    def get_features(self, train_folder, debug=False):
        pickled_folder = self.args.pickled_folder + "_{}_{}".format(self.args.bert_model, str(self.args.skip_no_ans))

        features_lst = []

        files = [f for f in os.listdir(train_folder) if f.endswith(".gz")]
        names = [f.split(".")[0] for f in files]
        print("Number of data sets:{}".format(len(files)))
        for filename in files:
            data_name = filename.split(".")[0]
            # Check whether pkl file already exists
            pickle_file_name = '{}.pkl'.format(data_name)
            pickle_file_path = os.path.join(pickled_folder, pickle_file_name)
            if os.path.exists(pickle_file_path):
                with open(pickle_file_path, 'rb') as pkl_f:
                    print("Loading {} file as pkl...".format(data_name))
                    features_lst.append(pickle.load(pkl_f))
            else:
                print("processing {} file".format(data_name))
                file_path = os.path.join(train_folder, filename)

                train_examples = read_squad_examples(file_path, debug=debug)

                train_features = convert_examples_to_features(
                    examples=train_examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.args.max_seq_length,
                    max_query_length=self.args.max_query_length,
                    doc_stride=self.args.doc_stride,
                    is_training=True,
                    skip_no_ans=self.args.skip_no_ans
                )

                features_lst.append(train_features)

                # Save feature lst as pickle (For reuse & fast loading)
                if not debug and self.args.rank == 0:
                    with open(pickle_file_path, 'wb') as pkl_f:
                        print("Saving {} file from pkl file...".format(data_name))
                        pickle.dump(train_features, pkl_f)

        return features_lst, names

    def get_iter(self, features_lst, args):
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_start_positions = []
        all_end_positions = []
        all_labels = []

        print("Dataset sizes")
        for i, train_features in enumerate(features_lst):
            all_input_ids.append(torch.tensor([f.input_ids for f in train_features], dtype=torch.long))
            all_input_mask.append(torch.tensor([f.input_mask for f in train_features], dtype=torch.long))
            all_segment_ids.append(torch.tensor([f.segment_ids for f in train_features], dtype=torch.long))

            start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

            all_start_positions.append(start_positions)
            all_end_positions.append(end_positions)
            all_labels.append(i * torch.ones_like(start_positions))
            print(i, start_positions.shape)

        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_input_mask = torch.cat(all_input_mask, dim=0)
        all_segment_ids = torch.cat(all_segment_ids, dim=0)
        all_start_positions = torch.cat(all_start_positions, dim=0)
        all_end_positions = torch.cat(all_end_positions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_labels)
        if args.distributed:
            train_sampler = DistributedSampler(train_data)
            data_loader = DataLoader(train_data, num_workers=args.workers, pin_memory=True,
                                     sampler=train_sampler, batch_size=args.batch_size)
        else:
            weights = make_weights_for_balanced_classes(all_labels.detach().cpu().numpy().tolist(), self.args.num_classes)
            weights = torch.DoubleTensor(weights)
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=None,
                                                      sampler=train_sampler, num_workers=args.workers,
                                                      worker_init_fn=self.set_random_seed(self.args.random_seed), pin_memory=True, drop_last=True)

        return data_loader, train_sampler

    def save_model(self, epoch, loss):
        loss = round(loss, 3)
        model_type = ("adv" if self.args.adv else "base")

        save_file = os.path.join(self.args.save_dir, "{}_{}_{:.3f}.pt".format(model_type, epoch, loss))
        save_file_config = os.path.join(self.args.save_dir, "{}_config_{}_{:.3f}.json".format(model_type, epoch, loss))

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        torch.save(model_to_save.state_dict(), save_file)
        model_to_save.config.to_json_file(save_file_config)

    def train(self):
        step = 1
        avg_loss = 0
        global_step = 1
        iter_lst = [self.get_iter(self.features_lst, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            self.model.train()
            start = time.time()
            batch_step = 1
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)

                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions, _ = batch

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)

                    input_ids = input_ids[:, :max_len].clone()
                    input_mask = input_mask[:, :max_len].clone()
                    seg_ids = seg_ids[:, :max_len].clone()
                    start_positions = start_positions.clone()
                    end_positions = end_positions.clone()

                    if self.args.use_cuda:
                        input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
                        input_mask = input_mask.cuda(self.args.gpu, non_blocking=True)
                        seg_ids = seg_ids.cuda(self.args.gpu, non_blocking=True)
                        start_positions = start_positions.cuda(self.args.gpu, non_blocking=True)
                        end_positions = end_positions.cuda(self.args.gpu, non_blocking=True)

                    loss = self.model(input_ids, seg_ids, input_mask, start_positions, end_positions)
                    loss = loss.mean()
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()

                    avg_loss = self.cal_running_avg_loss(loss.item() * self.args.gradient_accumulation_steps, avg_loss)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if epoch != 0 and i % 2000 == 0:
                        result_dict = self.evaluate_model(i)
                        for dev_file, f1 in result_dict.items():
                            print("GPU/CPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

                    global_step += 1
                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches),
                                avg_loss)
                    print(msg, end="\r")

            print("[GPU Num: {}, epoch: {}, Final loss: {:.4f}]".format(self.args.gpu, epoch, avg_loss))

            # save model
            if self.args.rank == 0:
                self.save_model(epoch, avg_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU/CPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def evaluate_model(self, epoch):
        # result directory
        result_file = os.path.join(self.args.result_dir, "dev_eval_{}.txt".format(epoch))
        fw = open(result_file, "a")
        result_dict = dict()
        for dev_file in self.dev_files:
            file_name = dev_file.split(".")[0]
            prediction_file = os.path.join(self.args.result_dir, "epoch_{}_{}.json".format(epoch, file_name))
            file_path = os.path.join(self.args.dev_folder, dev_file)
            metrics = eval_qa(self.model, file_path, prediction_file, args=self.args, tokenizer=self.tokenizer, batch_size=self.args.batch_size)
            f1 = metrics["f1"]
            fw.write("{} : {}\n".format(file_name, f1))
            result_dict[dev_file] = f1
        fw.close()

        return result_dict

    def test(self):
        pass

    @staticmethod
    def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss

    @staticmethod
    def set_random_seed(random_seed):
        if random_seed is not None:
            print("Set random seed as {}".format(random_seed))
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.set_num_threads(1)
            cudnn.benchmark = False
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')


class AdvTrainer(BaseTrainer):
    def __init__(self, args):
        super(AdvTrainer, self).__init__(args)

    def make_model_env(self, gpu, ngpus_per_node):
        if self.args.distributed:
            self.args.gpu = self.args.devices[gpu]
        else:
            self.args.gpu = 0

        if self.args.use_cuda and self.args.distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = DomainQA(self.args.bert_model, self.args.num_classes,
                              self.args.hidden_size, self.args.num_layers,
                              self.args.dropout, self.args.dis_lambda,
                              self.args.concat, self.args.anneal)

        if self.args.load_model is not None:
            print("Loading model from ", self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model, map_location=lambda storage, loc: storage))

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) * self.args.epochs * len(self.features_lst)

        qa_params = list(self.model.bert.named_parameters()) + list(self.model.qa_outputs.named_parameters())
        dis_params = list(self.model.discriminator.named_parameters())
        self.qa_optimizer = get_opt(qa_params, num_train_optimization_steps, self.args)
        self.dis_optimizer = get_opt(dis_params, num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model.cuda()
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def train(self):
        step = 1
        avg_qa_loss = 0
        avg_dis_loss = 0
        iter_lst = [self.get_iter(self.features_lst, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])

        correct = torch.zeros((6), dtype=torch.float)
        tp = torch.zeros((6), dtype=torch.float)
        fp = torch.zeros((6), dtype=torch.float)
        fn = torch.zeros((6), dtype=torch.float)
        correct_total = 0

        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            start = time.time()
            self.model.train()
            batch_step = 1
            data_len = 0
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)

                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions, labels = batch

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)

                    input_ids = input_ids[:, :max_len].clone()
                    input_mask = input_mask[:, :max_len].clone()
                    seg_ids = seg_ids[:, :max_len].clone()
                    start_positions = start_positions.clone()
                    end_positions = end_positions.clone()

                    if self.args.use_cuda:
                        input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
                        input_mask = input_mask.cuda(self.args.gpu, non_blocking=True)
                        seg_ids = seg_ids.cuda(self.args.gpu, non_blocking=True)
                        start_positions = start_positions.cuda(self.args.gpu, non_blocking=True)
                        end_positions = end_positions.cuda(self.args.gpu, non_blocking=True)

                    qa_loss = self.model(input_ids, seg_ids, input_mask,
                                         start_positions, end_positions, labels,
                                         dtype="qa",
                                         global_step=step)
                    qa_loss = qa_loss.mean()
                    qa_loss.backward()

                    # update qa model
                    avg_qa_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
                    self.qa_optimizer.step()
                    self.qa_optimizer.zero_grad()

                    # update discriminator
                    dis_loss, log_prob = self.model(input_ids, seg_ids, input_mask,
                                          start_positions, end_positions, labels, dtype="dis",
                                          global_step=step)
                    dis_loss = dis_loss.mean()
                    dis_loss.backward()
                    avg_dis_loss = self.cal_running_avg_loss(dis_loss.item(), avg_dis_loss)
                    self.dis_optimizer.step()
                    self.dis_optimizer.zero_grad()
                    step += 1
                    if epoch != 0 and i % 2000 == 0:
                        result_dict = self.evaluate_model(i)
                        for dev_file, f1 in result_dict.items():
                            print("GPU/CPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - QA loss: {:.4f}, DIS loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches),
                                avg_qa_loss, avg_dis_loss)

                    writer.add_scalar("QA_loss", avg_qa_loss, i)
                    writer.add_scalar("Discriminator_loss", avg_dis_loss, i)

                    print(msg, end="\r")

                    data_len += labels.shape[0]
                    onehot_labels = torch.nn.functional.one_hot(labels, num_classes=6).float()
                    onehot_pred = torch.nn.functional.one_hot((log_prob.argmax(dim=1).detach().cpu()),
                                                              num_classes=6).float()
                    correct_total += ((log_prob.argmax(dim=1).detach().cpu())==labels.detach().cpu()).float().sum()
                    correct += (onehot_pred == onehot_labels).sum(dim=0).float()
                    tp += ((onehot_pred.float() == 1) & (onehot_labels.float() == 1)).sum(dim=0).float()
                    fp += ((onehot_pred.float() == 1) & (onehot_labels.float() == 0)).sum(dim=0).float()
                    fn += ((onehot_pred.float() == 0) & (onehot_labels.float() == 1)).sum(dim=0).float()
                    writer.add_scalar("Accuracy", correct_total / data_len, i)
                    writer.add_scalars("Accuracy_by_class", summary_map(self.num_to_name, correct / data_len), i)
                    writer.add_scalars("True_positives", summary_map(self.num_to_name, tp / data_len), i)
                    writer.add_scalars("False_negatives", summary_map(self.num_to_name, fn / data_len), i)
                    writer.add_scalars("False_positives", summary_map(self.num_to_name, fp / data_len), i)
                    if i % 1000 == 0:
                        print(
                            "Accuracy total {}, by class {}, tp {}, fp {}, fn {}".format(correct_total / data_len,
                                                                                         correct / data_len,
                                                                                         tp / data_len, fp / data_len,
                                                                                         fn / data_len), end="\n")

            print("[GPU Num: {}, Epoch: {}, Final QA loss: {:.4f}, Final DIS loss: {:.4f}]"
                  .format(self.args.gpu, epoch, avg_qa_loss, avg_dis_loss))

            print(
                "Accuracy total {}, by class {}, tp {}, fp {}, fn {}".format(correct_total / data_len,
                                                                             correct / data_len,
                                                                             tp / data_len, fp / data_len,
                                                                             fn / data_len), end="\n")
            # save model
            if not self.args.distributed or self.args.rank == 0:
                self.save_model(epoch, avg_qa_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU/CPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def test(self):
        correct = torch.zeros((6), dtype=torch.float)
        tp = torch.zeros((6), dtype=torch.float)
        fp = torch.zeros((6), dtype=torch.float)
        fn = torch.zeros((6), dtype=torch.float)
        correct_total = 0
        step = 1
        data_len = 0
        iter_lst = [self.get_iter(self.features_lst, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        start = time.time()
        for data_loader, sampler in iter_lst:
            for i, batch in enumerate(data_loader, start=1):
                input_ids, input_mask, seg_ids, start_positions, end_positions, labels = batch

                # remove unnecessary pad token
                seq_len = torch.sum(torch.sign(input_ids), 1)
                max_len = torch.max(seq_len)

                input_ids = input_ids[:, :max_len].clone()
                input_mask = input_mask[:, :max_len].clone()
                seg_ids = seg_ids[:, :max_len].clone()
                start_positions = start_positions.clone()
                end_positions = end_positions.clone()

                if self.args.use_cuda:
                    input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask.cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids.cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.cuda(self.args.gpu, non_blocking=True)


                dis_loss, log_prob = self.model(input_ids, seg_ids, input_mask,
                                      start_positions, end_positions, labels, dtype="dis",
                                      global_step=step)

                #print(log_prob.shape, labels.shape)
                data_len += labels.shape[0]
                onehot_labels = torch.nn.functional.one_hot(labels, num_classes=6).float()
                onehot_pred = torch.nn.functional.one_hot((log_prob.argmax(dim=1).detach().cpu()), num_classes=6).float()
                correct_total += ((log_prob.argmax(dim=1).detach().cpu()) == labels.detach().cpu()).float().sum()
                correct += (onehot_pred == onehot_labels).sum(dim=0).float()
                tp += ((onehot_pred.float() == 1) & (onehot_labels.float() == 1)).sum(dim=0).float()
                fp += ((onehot_pred.float() == 1) & (onehot_labels.float() == 0)).sum(dim=0).float()
                fn += ((onehot_pred.float() == 0) & (onehot_labels.float() == 1)).sum(dim=0).float()

                msg = "{}/{} {} - ETA : {}" .format(i, num_batches, progress_bar(i, num_batches), eta(start, i, num_batches))
                writer.add_scalar("Accuracy", correct_total / data_len, i)
                writer.add_scalars("Accuracy_by_class", summary_map(self.num_to_name, correct / data_len), i)
                writer.add_scalars("True_positives", summary_map(self.num_to_name, tp / data_len), i)
                writer.add_scalars("False_negatives", summary_map(self.num_to_name, fn / data_len), i)
                writer.add_scalars("False_positives", summary_map(self.num_to_name, fp / data_len), i)
                if i % 1000 == 0:
                    print(
                        "Accuracy total {}, by class {}, tp {}, fp {}, fn {}".format(correct_total / data_len,
                                                                                     correct / data_len,
                                                                                     tp / data_len, fp / data_len,
                                                                                     fn / data_len), end="\n")
                    print(msg)
                else:
                    print(msg, end="\r")

        print(
            "Accuracy total {}, by class {}, tp {}, fp {}, fn {}".format(correct_total / data_len,
                                                                         correct / data_len,
                                                                         tp / data_len, fp / data_len,
                                                                         fn / data_len), end="\n")





