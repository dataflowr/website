@def sequence = ["loss-functions"]

# Module 3 - Loss functions for classification

**Table of Contents**

\toc


## Loss functions for classification

{{youtube_placeholder loss-functions}}

{{yt_tsp 0 0 Recap}}
{{yt_tsp 145 0 How to choose your loss?}}
{{yt_tsp 198 0 A probabilistic model for linear regression}}
{{yt_tsp 470 0 Gradient descent, learning rate, SGD}}
{{yt_tsp 690 0 Pytorch code for gradient descent}}
{{yt_tsp 915 0 A probabilistic model for logistic regression}}
{{yt_tsp 1047 0 Notations (information theory)}}
{{yt_tsp 1258 0 Likelihood for logistic regression}}
{{yt_tsp 1363 0 BCELoss}}
{{yt_tsp 1421 0 BCEWithLogitsLoss}}
{{yt_tsp 1537 0 Beware of the reduction parameter}}
{{yt_tsp 1647 0 Softmax regression}}
{{yt_tsp 1852 0 NLLLoss}}
{{yt_tsp 2088 0 Classification in pytorch}}
{{yt_tsp 2196 0 Why maximizing accuracy directly is hard?}}
{{yt_tsp 2304 0 Classification in deep learning}}
{{yt_tsp 2450 0 Regression without knowing the underlying model}}
{{yt_tsp 2578 0 Overfitting in polynomial regression}}
{{yt_tsp 2720 0 Validation set}}
{{yt_tsp 2935 0 Notion of risk and hypothesis space}}
{{yt_tsp 3280 0 estimation error and approximation error}}

## Slides and Notebook

- [slides](https://dataflowr.github.io/slides/module3.html)
- [notebook](https://github.com/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb) in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb) An explanation of underfitting and overfitting with polynomial regression.

## Minimal working examples

### [`BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
```python
import torch.nn as nn
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3,4,5)
target = torch.randn(3,4,5)
loss(m(input), target)
```

### [`NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) and [`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
```python
import torch.nn as nn
m = nn.LogSoftmax(dim=1)
loss1 = nn.NLLLoss()
loss2 = nn.CrossEntropyLoss()
C = 8
input = torch.randn(3,C,4,5)
target = torch.empty(3,4,5 dtype=torch.long).random_(0,C) 
assert loss1(m(input),target) == loss2(input,target)
```

## Quiz

To check you know your loss, you can do the [quizzes](https://dataflowr.github.io/quiz/module3.html)

