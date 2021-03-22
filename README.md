This is a modified version of the following work:
# Domain-agnostic Question-Answering with Adversarial Training

Code for our paper ["Domain-agnostic Question-Answering with Adversarial Training"](https://arxiv.org/abs/1910.09342) which is accepted by EMNLP-IJCNLP 2019 MRQA Workshop.

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/77993568-58faa800-7363-11ea-8900-f01ec9b4980d.png" />  
</p>

## Data Preparation

### Download the original data

- Download the data by running shell file.
- Then run the code. Preprocessed train data will be created before training (It will takes quite a long time)

```bash
$ cd data
$ ./download_data.sh
```

### (Optional) Download the pickled data (for fast data loading)

- Download the pickled data from this [link](https://drive.google.com/open?id=1-IHdLL4oLOI_Ur8ej-KUZ4kVGGuSKcJ2).

- Unzip the zipfile on the root directory.

```bash
.
├── ...
├── pickled_data_bert-base-uncased_False
│   ├── HotpotQA.pkl
│   ├── NaturalQuestions.pkl
│   ├── NewsQA.pkl
│   ├── SQuAD.pkl
│   ├── SearchQA.pkl
│   └── TriviaQA.pkl
└── ...

```

- **Arguments should be same as below if you use pickled data. If you want to change one of these two arguments.**

```bash
parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="Bert model")
parser.add_argument("--skip_no_ans", action='store_true', default=False, help="whether to exclude no answer example")
```

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

```bash
torch==1.1.0
pytorch-pretrained-bert>=0.6.2
json-lines>=0.5.0
```

Alternatively, you can replicate the environment from the mrqa_env.yml
conda env create -f mrqa_env.yml

## Model Training & Validation

```bash
$ python main.py --epochs 1 \
                --batch_size 32 \
                --lr 3e-5 --do_lower_case \
                --use_cuda --devices 0_1_2_3
                --do_valid --adv\
                --dis_lambda 0.01\
                --qa_emb cls\
                --loss ce
```

- If you are using uncased bert model, give the option `--do_lower_case`.
- If you want to do validation, give the option `--do_valid`.

The testing options are performed on the held-out subsets of the training datasets
- If you only want to run the test use the `--only_test` option.
- You can control the side of this subset with the `--train_split` parameter.

You can monitor the training by running
```tensorboard --logdir=runs```

There are two options for the loss functions of the discriminator: the original cross entropy loss and the new Wasserstein distance loss, this is controlled by the `--loss` argument.
Theere are three options for the embedding that is used for the discriminator: the CLS token (used in the original work), the pooler output or the concatenated average embeddings of the question and answer. Which one is used is controlled by the `--qa_emb` argument.

The validation F1 scores can be found in the `results/adv_{date}/dev_eval_{epoch}.txt` files.
The saved models in `save/adv_{date}` folders.


## Reference

```
@inproceedings{lee-etal-2019-domain,
    title={Domain-agnostic Question-Answering with Adversarial Training},
    author={Seanie Lee and Donggyu Kim and Jangwon Park},
    booktitle={Proceedings of the 2nd Workshop on Machine Reading for Question Answering},
    publisher={Association for Computational Linguistics},
    year={2019},
    url={https://www.aclweb.org/anthology/D19-5826},
}
```

## Original contributors

- Lee, Seanie (https://seanie12.github.io/)
- Kim, Donggyu (https://github.com/donggyukimc)
- Park, Jangwon (https://github.com/monologg)
