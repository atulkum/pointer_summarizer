pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*

Train with pointer generation + coverage loss enabled 
--------------------------------------------
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

Training with pointer generation enabled
--------------------------------------------

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


How to run training:
--------------------------------------------
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail
2) Run start_train.sh, you might need to change some path and parameters in data_util/config.py
3) For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh

Note:
* It is tested on pytorch 0.3 
* You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score



