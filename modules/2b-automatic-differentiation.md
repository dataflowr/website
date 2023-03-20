@def sequence = ["automatic-diff"]

# Module 2b - Automatic differentiation

**Table of Contents**

\toc


## Automatic differentiation

{{youtube_placeholder automatic-diff}}

{{yt_tsp 0 0 Recap}}
{{yt_tsp 40 0 A simple example (more in the practicals)}}
{{yt_tsp 224 0 Pytorch tensor: requires_grad field}}
{{yt_tsp 404 0 Pytorch backward function}}
{{yt_tsp 545 0 The chain rule on our example}}
{{yt_tsp 960 0 Linear regression}}
{{yt_tsp 1080 0 Gradient descent with numpy...}}
{{yt_tsp 1650 0 ... with pytorch tensors}}
{{yt_tsp 1890 0 Using autograd}}
{{yt_tsp 2075 0 Using a neural network (linear layer)}}
{{yt_tsp 2390 0 Using a pytorch optimizer}}
{{yt_tsp 2640 Backprop algorithm: how automatic differentiation works}}

## Slides and Notebook

- Automatic differentiation: a simple example [static notebook](/notebooks_md/02a_basics), [code (GitHub)](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb) in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)
- [notebook](https://github.com/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb) used in the video for the linear regression. If you want to open it in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb)
- [backprop slide](https://raw.githubusercontent.com/dataflowr/slides/master/backprop.pdf) (used for the practical below)

## Quiz

To check your understanding of automatic differentiation, you can do the [quizzes](https://dataflowr.github.io/quiz/module2.html)
## Practicals

![](https://dataflowr.github.io/notebooks/Module2/img/backprop3.png)

- [practicals](https://github.com/dataflowr/notebooks/blob/master/Module2/02_backprop.ipynb) in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02_backprop.ipynb) Coding backprop. <!-- [solution](https://forum.dataflowr.com/t/link-to-solution-2-simple-implementation-of-backprop/55) (forum login required) -->

## Challenge

Adapt your code to solve the following challenge:

![](https://dataflowr.github.io/notebooks/Module2/img/backprop4.png)

Some small modifications:
- First modification: we now generate points $(x_t,y_t)$ where $y_t= \exp(w^*\cos(x_t)+b^*)$, i.e $y^*_t$ is obtained by applying a deterministic function to $x_t$ with parameters $w^*$ and $b^*$. Our goal is still to recover the parameters $w^*$ and $b^*$ from the observations $(x_t,y_t)$.

- Second modification: we now generate points $(x_t,y_t)$ where $y_t= \exp(w^*\cos(p^*x_t)+b^*)$, i.e $y^*_t$ is obtained by applying a deterministic function to $x_t$ with parameters $p^*$, $w^*$ and $b^*$. Our goal is still to recover the parameters from the observations $(x_t,y_t)$.

## Bonus: 

- [JAX](https://jax.readthedocs.io/en/latest/index.html) implementation of the linear regression [notebook](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb) in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb) see [Module 2c](/modules/2c-jax) for more details.