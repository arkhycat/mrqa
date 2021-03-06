import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig
from utils import kl_coef


def wasserstein_dist(tensor_a, tensor_b):
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    cdf_loss = cdf_distance.mean()
    return cdf_loss

class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, bert_name_or_config, num_classes=6, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False, loss='ce', emb_disc='cls'):
        super(DomainQA, self).__init__()
        if isinstance(bert_name_or_config, BertConfig):
            self.bert = BertModel(bert_name_or_config)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.config = self.bert.config

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat or emb_disc == 'qa':
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102
        self.disc_loss = loss
        self.emb_disc = emb_disc

    # only for prediction
    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, token_type_ids, attention_mask,
                                      start_positions, end_positions, global_step)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss, logits = self.forward_discriminator(input_ids, token_type_ids, attention_mask, labels, start_positions, end_positions)
            return dis_loss, logits

        else:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def forward_qa(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions, global_step):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        log_prob = None
        if self.emb_disc == 'qa':
            q, a = self.get_avg_qa_emb(input_ids, sequence_output, start_positions, end_positions, token_type_ids)
            log_prob = self.discriminator(torch.cat((a, q), 1))
        elif self.emb_disc == 'pool':
            log_prob = self.discriminator(pooled_output)
        elif self.emb_disc == 'cls':
            cls_embedding = sequence_output[:, 0]
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
            else:
                hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation
            log_prob = self.discriminator(hidden.detach())

        criterion = None
        if self.disc_loss == 'wass':
            criterion = wasserstein_dist
        elif self.disc_loss == 'ce':
            criterion = nn.KLDivLoss(reduction="batchmean")

        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        kld = self.dis_lambda * criterion(log_prob, targets)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, token_type_ids, attention_mask, labels, start_positions, end_positions):
        with torch.no_grad():
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        log_prob = None
        if self.emb_disc == 'qa':
            q, a = self.get_avg_qa_emb(input_ids, sequence_output, start_positions, end_positions, token_type_ids)
            log_prob = self.discriminator(torch.cat((a, q), 1))
        elif self.emb_disc == 'pool':
            log_prob = self.discriminator(pooled_output)
        elif self.emb_disc == 'cls':
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            else:
                hidden = cls_embedding
            log_prob = self.discriminator(hidden.detach())

        if self.disc_loss == 'ce':
            criterion = nn.NLLLoss()
            loss = criterion(log_prob, labels)
        if self.disc_loss == 'wass':
            onehot = torch.FloatTensor(log_prob.shape).to(log_prob.device)
            onehot.zero_()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            loss = wasserstein_dist(log_prob, onehot)

        return loss, log_prob

    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == self.sep_id).sum(1)
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding

    def get_avg_qa_emb(self, input_ids, sequence_output, start_positions, end_positions, token_type_ids):
        answer_phrase_ids = torch.zeros_like(input_ids)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1) #1 at start position
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1) # -1 at end position
        answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).to(input_ids.device))
        # (get 1's at all the postions of the answer)
        answer_len = (answer_phrase_ids.sum(-1) + 1e-9).unsqueeze(-1) # b x 1
        answer_phrase_ids = answer_phrase_ids.unsqueeze(1) # b x 1 x seq
        avg_phrase_emb = answer_phrase_ids.matmul(sequence_output ).squeeze(1)
        answer = avg_phrase_emb / answer_len # b x 1 x hidden / b x 1

        question_mask = (1 - token_type_ids).unsqueeze(-1).float()
        question_len = question_mask.sum((-2, -1))
        question = question_mask.mul(sequence_output).sum(-2)
        question /= question_len.unsqueeze(-1)

        return question, answer
