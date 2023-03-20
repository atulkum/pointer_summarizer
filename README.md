[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/atulkum/pointer_summarizer)

pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*

1. [Train with pointer generation and coverage loss enabled](#train-with-pointer-generation-and-coverage-loss-enabled)
2. [Training with pointer generation enabled](#training-with-pointer-generation-enabled)
3. [How to run training](#how-to-run-training)
4. [Papers using this code](#papers-using-this-code)


## Train with pointer generation and coverage loss enabled 
After training for 100k iterations with coverage loss enabled (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3907 with confidence interval (0.3885, 0.3928)
rouge_1_recall: 0.4434 with confidence interval (0.4410, 0.4460)
rouge_1_precision: 0.3698 with confidence interval (0.3672, 0.3721)

ROUGE-2:
rouge_2_f_score: 0.1697 with confidence interval (0.1674, 0.1720)
rouge_2_recall: 0.1920 with confidence interval (0.1894, 0.1945)
rouge_2_precision: 0.1614 with confidence interval (0.1590, 0.1636)

ROUGE-l:
rouge_l_f_score: 0.3587 with confidence interval (0.3565, 0.3608)
rouge_l_recall: 0.4067 with confidence interval (0.4042, 0.4092)
rouge_l_precision: 0.3397 with confidence interval (0.3371, 0.3420)
```

![Alt text](learning_curve_coverage.png?raw=true "Learning Curve with coverage loss")

## Training with pointer generation enabled
After training for 500k iterations (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3500 with confidence interval (0.3477, 0.3523)
rouge_1_recall: 0.3718 with confidence interval (0.3693, 0.3745)
rouge_1_precision: 0.3529 with confidence interval (0.3501, 0.3555)

ROUGE-2:
rouge_2_f_score: 0.1486 with confidence interval (0.1465, 0.1508)
rouge_2_recall: 0.1573 with confidence interval (0.1551, 0.1597)
rouge_2_precision: 0.1506 with confidence interval (0.1483, 0.1529)

ROUGE-l:
rouge_l_f_score: 0.3202 with confidence interval (0.3179, 0.3225)
rouge_l_recall: 0.3399 with confidence interval (0.3374, 0.3426)
rouge_l_precision: 0.3231 with confidence interval (0.3205, 0.3256)
```
![Alt text](learning_curve.png?raw=true "Learning Curve with pointer generation")


## How to run training:
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail
2) Run start_train.sh, you might need to change some path and parameters in data_util/config.py
3) For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh

Note:

* In decode mode beam search batch should have only one example replicated to batch size
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/decode.py#L109
https://github.com/atulkum/pointer_summarizer/blob/master/data_util/batcher.py#L226

* It is tested on pytorch 0.4 with python 2.7
* You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score

## Papers using this code:
1) [Automatic Program Synthesis of Long Programs with a Learned Garbage Collector](http://papers.nips.cc/paper/7479-automatic-program-synthesis-of-long-programs-with-a-learned-garbage-collector) ***NeuroIPS 2018*** https://github.com/amitz25/PCCoder
2) [Automatic Fact-guided Sentence Modification](https://arxiv.org/abs/1909.13838) ***AAAI 2020*** https://github.com/darsh10/split_encoder_pointer_summarizer
3) [Resurrecting Submodularity in Neural Abstractive Summarization](https://arxiv.org/abs/1911.03014v1)
4) [StructSum: Summarization via Structured Representations](https://aclanthology.org/2021.eacl-main.220) ***EACL 2021***
5) [Concept Pointer Network for Abstractive Summarization](https://arxiv.org/abs/1910.08486) ***EMNLP'2019*** https://github.com/wprojectsn/codes
6) [PaddlePaddle version](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_summarization/pointer_summarizer)
7) [VAE-PGN based Abstractive Model in Multi-stage Architecture for Text Summarization](https://www.aclweb.org/anthology/W19-8664/) ***INLG2019***
8) [Clickbait? Sensational Headline Generation with Auto-tuned Reinforcement Learning](https://arxiv.org/abs/1909.03582)  ***EMNLP'2019*** https://github.com/HLTCHKUST/sensational_headline
9) [Abstractive Spoken Document Summarization using Hierarchical Model with Multi-stage Attention Diversity Optimization](http://www.interspeech2020.org/index.php?m=content&c=index&a=show&catid=354&id=1173) ***INTERSPEECH 2020***
10) [Nutribullets Hybrid: Multi-document Health Summarization](https://arxiv.org/abs/2104.03465) ***NAACL 2021***
11) [A Corpus of Very Short Scientific Summaries](https://aclanthology.org/2020.conll-1.12) ***CoNLL 2020***
12) [Towards Faithfulness in Open Domain Table-to-text Generation from an Entity-centric View](https://arxiv.org/abs/2102.08585) ***AAAI 2021***
13) [CDEvalSumm: An Empirical Study of Cross-Dataset Evaluation for Neural Summarization Systems](https://aclanthology.org/2020.findings-emnlp.329) ***Findings of EMNLP2020***
14) [A Study on Seq2seq for Sentence Compression in Vietnamese](https://aclanthology.org/2020.paclic-1.56) ***PACLIC 2020***
15) [Other Roles Matter! Enhancing Role-Oriented Dialogue Summarization via Role Interactions](https://aclanthology.org/2022.acl-long.182/) ***ACL 2022***

