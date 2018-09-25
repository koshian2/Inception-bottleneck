# Inception-bottleneck
Evaluating reduction of calculation by bottleneck layer (1x1 conv).

# Result
|    FW   | pattern | mode | α | val_acc |  (sd) | s / epoch | (sd) |
|:-------:|:--------:|:----:|:-:|--------:|------:|----------:|-----:|
|  Keras  |     A    |   1  | - |  88.67% | 0.21% |     76.01 | 1.46 |
|  Keras  |     B    |   2  | 2 |  90.48% | 0.18% |     79.77 | 2.56 |
|  Keras  |     C    |   2  | 4 |  90.33% | 0.24% |     68.30 | 0.81 |
|  Keras  |     D    |   3  | 2 |  90.88% | 0.28% |     94.76 | 2.08 |
|  Keras  |     E    |   3  | 4 |  90.33% | 0.21% |     84.47 | 1.19 |
| PyTorch |     A    |   1  | - |  89.37% | 0.14% |     46.75 | 0.26 |
| PyTorch |     B    |   2  | 2 |  90.85% | 0.13% |     45.58 | 0.16 |
| PyTorch |     C    |   2  | 4 |  90.21% | 0.29% |     40.39 | 0.36 |
| PyTorch |     D    |   3  | 2 |  90.83% | 0.09% |     48.53 | 0.40 |
| PyTorch |     E    |   3  | 4 |  89.90% | 0.22% |     41.47 | 0.21 |

* Tried 5 times per pattern.
* val_acc, (sd) are maximum of validation accuracy's median and standard-deviation.
* s / epoch is training time(seconds) per epoch excluding 1st epoch, and took median and standard-deviation.
* Trained by google colab.

# Model summary
Evaluate simple inception module like this:
![](https://github.com/koshian2/Inception-bottleneck/blob/master/images/bottleneck_06.png)  
Details are a little bit changed from [the original inception](https://arxiv.org/pdf/1409.4842.pdf) paper.

Replaced AlexNet's Conv2D with this inception module:

|  module | pixels | input_ch | output_ch |
|:-----------:|-------:|---------------:|---------------:|
| Inception 1 |     32 |              3 |             96 |
|  Pooling 1  |     16 |             96 |             96 |
| Inception 2 |     16 |             96 |            256 |
|  Pooling 2  |      8 |            256 |            256 |
| Inception 3 |      8 |            256 |            384 |
| Inception 4 |      8 |            384 |            384 |
| Inception 5 |      8 |            384 |            256 |

There are 3 modes.
1. No-bottleneck(output_ch=f)
2. Bottle-neck(output_ch=f/α) -> Conv(output_ch=f)
3. Bottle-neck(output_ch=f/α) -> Conv(output_ch=f/α) -> Bottle-neck(output_ch=f)

Changed the value of α to 2 and 4 and made 5 patterns: (# pseudo calculation is calculated by [this](https://github.com/koshian2/Inception-bottleneck/blob/master/utils/calc_inception_flops.py))

| pattern | mode | α |     # pseudo calculation | # params | calc/params |
|:--------:|:------:|:-:|------------:|---------------:|------:|
|     A    |    1   | - | 231,091,200 |      2,960,874 |  78.0 |
|     B    |    2   | 2 |  87,118,848 |        839,838 | 103.7 |
|     C    |    2   | 4 |  50,042,880 |        493,818 | 101.3 |
|     D    |    3   | 2 |  61,273,088 |        612,488 | 100.0 |
|     E    |    3   | 4 |  28,986,368 |        308,318 |  94.0 |

# More details(Japanese)
https://qiita.com/koshian2/items/031b6a335d0d217e4c4c
