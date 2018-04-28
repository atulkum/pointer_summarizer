pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*

Train with pointer generation + coverage loss enabled 
--------------------------------------------
After training for 100k iterations with coverage loss enabled (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3829 with confidence interval (0.3807, 0.3853)
rouge_1_recall: 0.4199 with confidence interval (0.4175, 0.4227)
rouge_1_precision: 0.3745 with confidence interval (0.3718, 0.3772)

ROUGE-2:
rouge_2_f_score: 0.1666 with confidence interval (0.1644, 0.1689)
rouge_2_recall: 0.1821 with confidence interval (0.1797, 0.1846)
rouge_2_precision: 0.1638 with confidence interval (0.1615, 0.1660)

ROUGE-l:
rouge_l_f_score: 0.3514 with confidence interval (0.3492, 0.3537)
rouge_l_recall: 0.3850 with confidence interval (0.3827, 0.3877)
rouge_l_precision: 0.3441 with confidence interval (0.3414, 0.3466)
```

![Alt text](learning_curve_coverage.png?raw=true "Learning Curve with coverage loss")

You can download the model [here](https://drive.google.com/open?id=1QqSaxcJGllVPSFea2c2iCV5_dtjJijVe).

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

You can download the model [here](https://drive.google.com/open?id=1kiarI44mVZCmadqgTnToo1jG-mRCzMaB).


How to run training:
--------------------------------------------
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail
2) Run start_train.sh, you might need to change some path and parameters in data_util/config.py
3) For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh

Note:
* It is tested on pytorch 0.3 
* You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score



