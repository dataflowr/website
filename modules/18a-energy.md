@def title = "energy"
@def hasmath = true


# Module 18a - Denoising Score Matching for Energy Based Models

This module is based on the work: [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288) by Yang Sond and Diederik P. Kingma (2021). 

~~~<img src="../extras/diffusions/energy.png"
           style="width: 820px; height: auto; display: inline">
~~~

**Table of Contents**

\toc

## Theory for Energy-Based Models (EBM)

The density given by an EBM is:
\begin{eqnarray*}
p_{\theta}(x) = \frac{\exp(-E_\theta(x))}{Z_\theta},
\end{eqnarray*}
where $E_\theta:\mathbb{R}^d \to \mathbb{R}$ and $Z_\theta=\int \exp(-E_\theta(x)) dx$.

Given samples $x_1,\dots, x_N$ in $\mathbb{R}^d$, we want to find the parameter $\theta$ maximizing the log-likelihood $\max_\theta \sum_{i=1}^N \log p_{\theta}(x_i)$. Since $Z_\theta$ is a function of $\theta$, evaluation and differentiation of $\log p_{\theta}(x)$ w.r.t. $\theta$ involves a typically intractable integral.

### Maximum Likelihood Training with MCMC

We can estimate the gradient of the log-likelihood with MCMC approaches:
\begin{eqnarray*}
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x)-\nabla_\theta 
\log Z_\theta.
\end{eqnarray*}
The first term is simple to compute (with automatic differentiation).

**Maths: computing $\nabla_\theta \log Z_\theta$**
We have:
\begin{eqnarray*}
\nabla_\theta \log Z_\theta = \mathbb{E}_{p_{\theta}(x)}\left[-\nabla_\theta E_\theta(x)\right] \left(= \int p_{\theta}(x) \left[-\nabla_\theta E_\theta(x)\right] dx \right).
\end{eqnarray*}
**Proof:**
\begin{eqnarray*}
\nabla_\theta \log Z_\theta &=& \frac{\nabla_\theta Z_\theta}{Z_\theta}\\
&=& \frac{1}{Z_\theta} \int \nabla_\theta \exp (-E_\theta(x))dx\\
&=& \frac{-1}{Z_\theta} \int \nabla_\theta E_\theta(x) \exp (-E_\theta(x))dx\\
&=& \mathbb{E}_{p_{\theta}(x)}\left[-\nabla_\theta E_\theta(x)\right]
\end{eqnarray*}

Thus, we can obtain an unbiased one-sample Monte Carlo estimate of the log-likelihood gradient by
\begin{eqnarray*}
\nabla_\theta \log Z_\theta \approx -\nabla_\theta E_\theta(\tilde{x}),
\end{eqnarray*}
with $\tilde{x}\sim p_\theta(x)$, i.e. a random sample from the distribution given by the EBM. Therefore, we need to draw random samples from the model. As explained during the course, this can be done using Langevin MCMC. First note that the gradient of the log-probability w.r.t. $x$ (which is the score) is easy to calculate:
\begin{eqnarray*}
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x) \text{ since }  \nabla_x \log Z_\theta = 0.
\end{eqnarray*}
Hence, in this case, Langevin MCMC is given by:
\begin{eqnarray*}
x_t = x_{t-1} - \epsilon \nabla_x E_\theta(x_{t-1}) +\sqrt{2\epsilon}z_t, 
\end{eqnarray*}
where $z_t\sim \mathcal{N}(0,I)$. When $\epsilon\to 0$ and $t\to \infty$, $x_t$ will be distributed as $p_\theta(x)$ (under some regularity conditions).

In this homework, we will consider an alternative learning procedure.

### Score Matching

The score (which was used in Langevin MCMC above) is defined as $$ s_\theta(x) = \nabla_x\log p_\theta(x) = -\nabla_x E_\theta(x) = -\left( \frac{\partial E_\theta(x)}{\partial x_1},\dots, \frac{\partial E_\theta(x)}{\partial x_d}\right).$$

If $p(x)$ denote the (unknown) data distribution, the basic score matching objective minimizes:
$$
\mathbb{E}_{p(x)} \|\nabla_x \log p(x) - s_\theta(x)\|^2.
$$


The problem with this objective is that we cannot compute $\nabla_x \log p(x)$ as $p(x)$ is unknown. We can only compute (approximate) averages with respect to $p(x)$ with empirical averages.
Fortunately, we can solve this issue as we have:
$$
\mathbb{E}_{p(x)} \|\nabla_x \log p(x) - s_\theta(x)\|^2 = c + \mathbb{E}_{p(x)}\left[ \sum_{i=1}^d\left ( \frac{\partial E_\theta(x)}{\partial x_i}\right)^2+2\frac{\partial^2 E_\theta(x)}{\partial x^2_i}\right],
$$
where $c$ is a constant (not depending on $\theta$).

**Proof:**
\begin{eqnarray*}
\mathbb{E}_{p(x)} \|\nabla_x \log p(x) - s_\theta(x)\|^2 &=&\mathbb{E}_{p(x)} \|\nabla_x \log p(x) \|^2 +\mathbb{E}_{p(x)} \| s_\theta(x)\|^2 - 2 \mathbb{E}_{p(x)} \langle \nabla_x \log p(x) , s_\theta(x)\rangle\\
&=& c + \mathbb{E}_{p(x)}\left[ \sum_{i=1}^d\left ( \frac{\partial E_\theta(x)}{\partial x_i}\right)^2\right] - 2 \int p(x)  \langle \frac{\nabla_x p(x)}{p(x)} , s_\theta(x)\rangle dx\\
&=& c + \mathbb{E}_{p(x)}\left[ \sum_{i=1}^d\left ( \frac{\partial E_\theta(x)}{\partial x_i}\right)^2\right] + 2\int p(x) \nabla_x \cdot s_\theta(x) dx,
\end{eqnarray*}
by integration by parts where for a vector valued function $v(x_1,x_2,x_3)$ $\nabla_x \cdot v = \frac{\partial v_1}{\partial x_1} + \frac{\partial v_2}{\partial x_2}+ \frac{\partial v_3}{\partial x_3}$. The statement follows.

### Denoising Score Matching

There are several drawbacks about the score matching approach: computing the trace of the Hessian is expensive and scores will not be accurately estimated in low-density regions, see [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/#naive-score-based-generative-modeling-and-its-pitfalls)

Denoising score matching is an elegant and scalable solution. Consider the random variable $Y = X+\sigma Z$, where $X\sim p(x)$ and $Z\sim\mathcal{N}(0,I)$. We denote by $p^\sigma(y)$ the distribution of $Y$ so that we have:
$$
\nabla_y\log p^\sigma(y) = -\frac{1}{\sigma}\mathbb{E}\left[ Z |Y=y\right] = -\frac{1}{\sigma}\mathbb{E}\left[ Z |X+\sigma Z=y\right].
$$
**Proof:**
\begin{eqnarray*}
\nabla_y\log p^\sigma(y) = \frac{\nabla_y p^\sigma(y)}{p^\sigma(y)}
\end{eqnarray*}
We denote by $\varphi$ the density of $\mathcal{N}(0,\sigma^2 I)$. We have $p^\sigma(y) = \int p(x) \varphi(y-x) dx$ so that using the fact that $\nabla_z \varphi(z) = -\frac{z}{\sigma^2} \varphi(z)$, we get
\begin{eqnarray*}
\nabla_y p^\sigma(y) &=& \int p(x) \nabla_y \varphi(y-x) dx\\
&=& \int p(x) \frac{-(y-x)}{\sigma^2} \varphi(y-x) dx \\
&=& -\frac{1}{\sigma}\mathbb{E}\left[ \frac{Y-X}{\sigma} |Y=y\right]\\
&=& -\frac{1}{\sigma}\mathbb{E}\left[ Z |Y=y\right]
\end{eqnarray*}

The denoising score matching objective is now
$$
\mathbb{E}_{p^\sigma(y)}\|\nabla_y \log p^\sigma(y) - s_\theta(y)\|^2,
$$
that we will minimize thanks to a gradient descent in the parameter $\theta$.

In practice, we use the following relation:
$$
\mathbb{E}_{p^\sigma(y)}\|\nabla_y \log p^\sigma(y) - s_\theta(y)\|^2 = \mathbb{E}\left\| \frac{Z}{\sigma}+s_\theta(X+\sigma Z)\right\|^2-C
$$
where $C$ does not depend on $\theta$ (made explicit below).

**Proof:**
We have
\begin{eqnarray*}
\mathbb{E}_{p^\sigma(y)}\|\nabla_y \log p^\sigma(y) - s_\theta(y)\|^2 &=& \mathbb{E} \left[\left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right] +s_\theta(Y)\right\|^2\right]\\
&=& \mathbb{E} \left[\left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right]\right\|^2 +  \left\|s_\theta(Y)\right\|^2 + 2 \left\langle \mathbb{E} \left[\frac{Z}{\sigma} | Y\right], s_\theta(Y)\right\rangle \right]\\
&=& \mathbb{E} \left[\left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right]\right\|^2 \right] + \mathbb{E} \left[ \mathbb{E} \left[ \left\|s_\theta(Y)\right\|^2 + 2 \left\langle \frac{Z}{\sigma},  s_\theta(Y)\right\rangle | Y \right]\right]\\
&=& \mathbb{E} \left[\left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right]\right\|^2 \right] + \mathbb{E} \left[  \left\|s_\theta(Y)\right\|^2 + 2 \left\langle \frac{Z}{\sigma},  s_\theta(Y)\right\rangle \right]\\
&=& \mathbb{E} \left[\left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right]\right\|^2 \right] + \mathbb{E} \left[  \left\|s_\theta(Y) + \frac{Z}{\sigma} \right\|^2 \right] - \mathbb{E} \left[ \left\|\frac{Z}{\sigma}\right\|^2\right]\\
&=& \mathbb{E}\left\| \frac{Z}{\sigma}+s_\theta(X+\sigma Z)\right\|^2 - \mathbb{E} \left[ \left\|\frac{Z}{\sigma}\right\|^2 - \left\| \mathbb{E} \left[\frac{Z}{\sigma} | Y\right]\right\|^2 \right].
\end{eqnarray*}

Hence, in practice, we will minimize the (random) loss:
$$
\ell(\theta; x_1,\dots, x_N) = \frac{1}{N} \sum_{i=1}^N \left\| \frac{z_i}{\sigma}+s_\theta(x_i+\sigma z_i)\right\|^2,
$$
where the $z_i$ are iid Gaussian. As the dataset is too large, we will run SGD algorithm, i.e. make batches and use automatic differentiation to get the gradient w.r.t. $\theta$ over each batch.

## Code for Energy Based Models

- Denoising Score Matching for Energy Based Models for a simple case [Denoising\_Score\_Matching\_Energy\_Model.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/Denoising_Score_Matching_Energy_Model_empty.ipynb). Here is the corresponding solution: [solution](https://github.com/dataflowr/notebooks/blob/master/Module18/Denoising_Score_Matching_Energy_Model_sol.ipynb) 





