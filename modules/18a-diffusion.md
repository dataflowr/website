@def title = "ddpm"
@def hasmath = true


# Module 18a - Denoising Diffusion Probabilistic Models

This module presents the work: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Jonathan Ho, Ajay Jain, Pieter Abbeel (2020). It starts with a description of the algorithm, then provides some notebooks to implement it on MNIST and CIFAR10 and finishes with some technical details.

**Table of Contents**

\toc
## Algorithm 

~~~<img src="../extras/diffusions/ddpm.png"
           style="width: 620px; height: auto; display: inline">
~~~

### Forward diffusion process

Given a schedule $\beta_1<\beta_2<\dots <\beta_T$,
\begin{align*}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t I)\\
q(x_{1:T}|x_0) &= \prod_{t=1}^T q(x_t|x_{t-1})
\end{align*}

We define $\alpha_t = 1-\beta_t$ and $\overline{\alpha_t} = \prod_{i=1}^t\alpha_i$, then we have
\begin{align*}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1},\text{ with }\epsilon_{t-1}\sim\mathcal{N}(0,I)\\
&= \sqrt{\alpha_t\alpha_{t-1}} x_{t-2} +\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2}+\sqrt{1-\alpha_t}\epsilon_{t-1}\\
&= \sqrt{\alpha_t\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\tilde{\epsilon}_{t}
\end{align*}
Hence, we have
\begin{align*}
x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon
\end{align*}
```python
class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, 
            beta_start=0.0001, beta_end=0.02, device=device):
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, 
                num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = 
                self.alphas_cumprod ** 0.5 
        self.sqrt_one_minus_alphas_cumprod = 
            (1 - self.alphas_cumprod) ** 0.5 

    def add_noise(self, x_start, noise, timesteps):
        # The forward process
        # x_start and noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * noise

    def reverse(self, x, t):
        # The network estimates the noise added
        return self.network(x, t)
```
### Approximating the reversed diffusion 

Note that the law $q(x_{t-1}|x_t,x_0)$ is explicit:
\begin{align*}
q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\mu(x_t,x_0), \gamma_t I), 
\end{align*}
with
\begin{align*}
\mu(x_t,x_0) &= \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}x_t + \frac{\sqrt{\overline{\alpha}_{t-1}\beta_t}}{1-\overline{\alpha}_{t}}x_0\\
\gamma_t &= \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_t
\end{align*}
but we know that $x_0 = 1/\sqrt{\overline{\alpha}_t}\left( x_t-\sqrt{1-\overline{\alpha}_t}\epsilon\right)$, hence we have
\begin{align*}
\mu(x_t,x_0) &= \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon\right) = \mu(x_t,t),
\end{align*}
where we removed the dependence in $x_0$ and replace it with a dependence in $t$.

The idea is to approximate $q(x_{t-1}|x_t)$ by a neural network according to:
\begin{align*}
p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,t), \beta_t I)
\end{align*}
and we approximate $q(x_{0:T})$ by
\begin{align*}
p(x_{0:T}) = p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t),
\end{align*}
where $p(x_T) \sim \mathcal{N}(0,I)$. Note that the variance parameter is fixed to $\beta_t$ which is the forward variance (mainly for simplicity, variations have been proposed).

The neural network is trained by maximizing the usual Variational bound:
\begin{align*}
\mathbb{E}_{q(x_0)} \ln p_{\theta}(x_0) &\geq \mathbb{E}_{q(x_{0:T})}\left[ \ln\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]\\
&=\mathbb{E}_q\left[ \text{KL}\left( q(x_T|x_0)\|p(x_T)\right)+\sum_{t=2}^T\text{KL}\left( q(x_{t-1}|x_t,x_0)\|p_{\theta}(x_{t-1}|x_t)\right)-\ln p_{\theta}(x_0|x_1)\right]\\
&= L_T +\sum_{t=2}^T L_{t-1}+L_0.
\end{align*}
Note that $L_T$ does not depend on $\theta$ and for the other terms, they correspond to a KL between Gaussian distributions with an explicit expression:
\begin{align*}
L_{t-1} = \mathbb{E}_q\left[ \frac{1}{2\beta_t^2}\|\mu_\theta(x_t,t) -\mu(x_t,t)\|^2\right]
\end{align*}
Now, we make the change of variable:
\begin{align*}
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right),
\end{align*}
so that we have
\begin{align*}
\|\mu_\theta(x_t,t) -\mu(x_t,t)\|^2 = \frac{(1-\alpha_t)^2}{1-\overline{\alpha}_t}\|\epsilon - \epsilon_\theta(\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon, t)\|^2
\end{align*}
Empirically, the prefactor is removed in the loss and instead of summing over all $t$, we average over a random $\tau\in [0,T-1]$, so that the loss is finally:
\begin{align*}
\ell(\theta) = \mathbb{E}_\tau\mathbb{E}_\epsilon \left[ \|\epsilon - \epsilon_\theta(\sqrt{\overline{\alpha}_\tau}x_0 + \sqrt{1-\overline{\alpha}_\tau}\epsilon, \tau)\|^2\right]
\end{align*}

```python
    # inside the training loop
    for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```


### Sampling

For sampling, we need to simulate the reversed diffusion (Markov chain) starting from $x_T\sim \mathcal{N}(0,I)$ and then:
\begin{align*}
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sqrt{\beta_t}\epsilon,\text{ with } \epsilon\sim\mathcal{N}(0,I).
\end{align*}
```python
    # inside Module DDPM
    def step(self, model_output, timestep, sample):
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/
                self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)
        pred_prev_sample = 
            coef_first_t*(sample-coef_eps_t*model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise)
            
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
```
## Summary: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) 
(J. Ho, A. Jain, P. Abbeel 2020)

~~~<img src="../extras/diffusions/ddpm.png"
           style="width: 620px; height: auto; display: inline">
~~~
@@colbox-blue
Given a schedule $\beta_1<\beta_2<\dots <\beta_T$, the **forward diffusion process** is defined by:
$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t I)$ and $q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$.

With $\alpha_t = 1-\beta_t$ and $\overline{\alpha_t} = \prod_{i=1}^t\alpha_i$, we see that, with $\epsilon\sim\mathcal{N}(0,I)$:
\begin{align*}
x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon.
\end{align*}
The law $q(x_{t-1}|x_t,\epsilon)$ is explicit: $q(x_{t-1}|x_t,\epsilon) = \mathcal{N}(x_{t-1};\mu(x_t,\epsilon,t), \gamma_t I)$ with,
\begin{align*}
\mu(x_t,\epsilon, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon\right)\text{ and, }
\gamma_t = \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_t
\end{align*}
@@

@@colbox-blue
**Training**: to approximate **the reversed diffusion** $q(x_{t-1}|x_t)$ by a neural network given by $p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,t), \beta_t I)$ and $p(x_T) \sim \mathcal{N}(0,I)$, we maximize the usual Variational bound:
\begin{align*}
\mathbb{E}_{q(x_0)} \ln p_{\theta}(x_0) &\geq L_T +\sum_{t=2}^T L_{t-1}+L_0 \text{ with, }L_{t-1} = \mathbb{E}_q\left[ \frac{1}{2\sigma_t^2}\|\mu_\theta(x_t,t) -\mu(x_t,\epsilon,t)\|^2\right].
\end{align*}
With the change of variable:
\begin{align*}
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right),
\end{align*}
ignoring the prefactor and sampling $\tau$ instead of summing over all $t$, the loss is finally:
\begin{align*}
\ell(\theta) = \mathbb{E}_\tau\mathbb{E}_\epsilon \left[ \|\epsilon - \epsilon_\theta(\sqrt{\overline{\alpha}_\tau}x_0 + \sqrt{1-\overline{\alpha}_\tau}\epsilon, \tau)\|^2\right]
\end{align*}
@@

@@colbox-blue
**Sampling**: to simulate the reversed diffusion with the learned $\epsilon_\theta(x_t,t)$ starting from $x_T\sim \mathcal{N}(0,I)$, iterate for $t=T,\dots, 1$:
\begin{align*}
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sqrt{\beta_t}\epsilon,\text{ with } \epsilon\sim\mathcal{N}(0,I).
\end{align*}
@@

## Implementation
  
![](../extras/diffusions/mnist_diffusion.gif)
### MNIST

The training of this notebook on colab takes approximately 20 minutes.

- [ddpm\_nano\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb) is the notebook where you code the DDPM algorithm (a simple UNet is provided for the network $\epsilon_\theta(x,t)$), its training and the sampling. You should get results like this:

![](../extras/diffusions/mnist_result.png)

- Here is the corresponding solution: [ddpm\_nano\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_sol.ipynb)


### CIFAR10

The training of this notebook on colab takes approximately 20 minutes (so do not expect high-quality pictures!). Still, after finetuning on specific classes, we see that the model learns features of the class.

~~~<img src="../extras/diffusions/diffusion_finetuning.png"
           style="width: 620px; height: auto; display: inline">
~~~

- [ddpm\_micro\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)

With a bit more training (100 epochs), you can get results like this:

![](../extras/diffusions/ships.png)

![](../extras/diffusions/horses.png)

![](../extras/diffusions/trucks.png)


## Technical details

Note that the Denoising Diffusion Probabilistic Model is the same for MNIST and CIFAR10, we only change the UNet learning to reverse the noise. For CIFAR10, we adapt the UNet provided in [Module 9b](../9b-unet). Indeed, you can still use the code provided here for DDPM with other architectures like more complex ones with self-attention like this [Unet](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L271) coded by [lucidrains](https://github.com/lucidrains) which is the one used in the original paper.

In the paper, the authors used Exponential Moving Average (EMA) on model parameters with a decay factor of $0.999$. This is not implemented here to keep the code as simple as possible.
