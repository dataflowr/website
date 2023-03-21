@def sequence = ["jax"]

# Module 2c - Automatic differentiation: VJP and intro to JAX

**Table of Contents**

\toc

# Autodiff and Backpropagation


## Jacobian

Let $\mathbf{f}:\mathbb{R}^n\to \mathbb{R}^m$, we define its Jacobian as:
\begin{align*}
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = J_{\mathbf{f}}(\mathbf{x}) &= \left( \begin{array}{ccc}
\frac{\partial f_1}{\partial x_1}&\dots& \frac{\partial f_1}{\partial x_n}\\
\vdots&&\vdots\\
\frac{\partial f_m}{\partial x_1}&\dots& \frac{\partial f_m}{\partial x_n}
\end{array}\right)\\
&=\left( \frac{\partial \mathbf{f}}{\partial x_1},\dots, \frac{\partial \mathbf{f}}{\partial x_n}\right)\\
&=\left(
\begin{array}{c}
\nabla f_1(\mathbf{x})^T\\
\vdots\\
\nabla f_m(x)^T
\end{array}\right)
\end{align*}

Hence the Jacobian $J_{\mathbf{f}}(\mathbf{x})\in \mathbb{R}^{m\times n}$ is a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ such that for $\mathbf{x},\mathbf{v} \in \mathbb{R}^n$ and $h\in \mathbb{R}$:
\begin{align*}
\mathbf{f}(\mathbf{x}+h\mathbf{v}) = \mathbf{f}(\mathbf{x}) + h J_{\mathbf{f}}(\mathbf{x})\mathbf{v} +o(h).
\end{align*}
The term $J_{\mathbf{f}}(\mathbf{x})\mathbf{v}\in \mathbb{R}^m$ is a Jacobian Vector Product (**JVP**), corresponding to the interpretation where the Jacobian is the linear map: $J_{\mathbf{f}}(\mathbf{x}):\mathbb{R}^n \to \mathbb{R}^m$, where $J_{\mathbf{f}}(\mathbf{x})(\mathbf{v})=J_{\mathbf{f}}(\mathbf{x})\mathbf{v}$.

## Chain composition

In machine learning, we are computing gradient of the loss function with respect to the parameters. In particular, if the parameters are high-dimensional, the loss is a real number. Hence, consider a real-valued function $\mathbf{f}:\mathbb{R}^n\stackrel{\mathbf{g}_1}{\to}\mathbb{R}^m \stackrel{\mathbf{g}_2}{\to}\mathbb{R}^d\stackrel{h}{\to}\mathbb{R}$, so that $\mathbf{f}(\mathbf{x}) = h(\mathbf{g}_2(\mathbf{g}_1(\mathbf{x})))\in \mathbb{R}$. We have
\begin{align*}
\underbrace{\nabla\mathbf{f}(\mathbf{x})}_{n\times 1}=\underbrace{J_{\mathbf{g}_1}(\mathbf{x})^T}_{n\times m}\underbrace{J_{\mathbf{g}_2}(\mathbf{g}_1(\mathbf{x}))^T}_{m\times d}\underbrace{\nabla h(\mathbf{g}_2(\mathbf{g}_1(\mathbf{x})))}_{d\times 1}.
\end{align*}
To do this computation, if we start from the right so that we start with a matrix times a vector to obtain a vector (of size $m$) and we need to make another matrix times a vector, resulting in $O(nm+md)$ operations. If we start from the left with the matrix-matrix multiplication, we get $O(nmd+nd)$ operations. Hence we see that as soon as $m\approx d$, starting for the right is much more efficient. Note however that doing the computation from the right to the left requires keeping in memory the values of $\mathbf{g}_1(\mathbf{x})\in\mathbb{R}^m$, and $\mathbf{x}\in \mathbb{R}^n$.

**Backpropagation** is an efficient algorithm computing the gradient "from the right to the left", i.e. backward. In particular, we will need to compute quantities of the form: $J_{\mathbf{f}}(\mathbf{x})^T\mathbf{u} \in \mathbb{R}^n$ with $\mathbf{u} \in\mathbb{R}^m$ which can be rewritten $\mathbf{u}^T J_{\mathbf{f}}(\mathbf{x})$ which is a Vector Jacobian Product (**VJP**), correponding to the interpretation where the Jacobian is the linear map: $J_{\mathbf{f}}(\mathbf{x}):\mathbb{R}^n \to \mathbb{R}^m$, composed with the linear map $\mathbf{u}:\mathbb{R}^m\to \mathbb{R}$ so that $\mathbf{u}^TJ_{\mathbf{f}}(\mathbf{x}) = \mathbf{u} \circ J_{\mathbf{f}}(\mathbf{x})$.

**example:** let $\mathbf{f}(\mathbf{x}, W) = \mathbf{x} W\in \mathbb{R}^b$ where $W\in \mathbb{R}^{a\times b}$ and $\mathbf{x}\in \mathbb{R}^a$. We clearly have
$$
J_{\mathbf{f}}(\mathbf{x}) = W^T.
$$
Note that here, we are slightly abusing notations and considering the partial function $\mathbf{x}\mapsto \mathbf{f}(\mathbf{x}, W)$. To see this, we can write $f_j = \sum_{i}x_iW_{ij}$ so that 
$$
\frac{\partial \mathbf{f}}{\partial x_i}= \left( W_{i1}\dots W_{ib}\right)^T
$$
Then recall from definitions that
$$
J_{\mathbf{f}}(\mathbf{x}) = \left( \frac{\partial \mathbf{f}}{\partial x_1},\dots, \frac{\partial \mathbf{f}}{\partial x_n}\right)=W^T.
$$
Now we clearly have
$$
J_{\mathbf{f}}(W) = \mathbf{x} \text{ since, } \mathbf{f}(\mathbf{x}, W+\Delta W) = \mathbf{f}(\mathbf{x}, W) + \mathbf{x} \Delta W.
$$
Note that multiplying $\mathbf{x}$ on the right is actually convenient when using broadcasting, i.e. we can take a batch of input vectors of shape $\text{bs}\times a$ without modifying the math above. 

## Implementation

In PyTorch, `torch.autograd` provides classes and functions implementing automatic differentiation of arbitrary scalar-valued functions. To create a custom [autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function), subclass this class and implement the `forward()` and `backward()` static methods. Here is an example:
```python
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
# Use it by calling the apply method:
output = Exp.apply(input)
```
You can have a look at [Module 2b](https://dataflowr.github.io/website/modules/2b-automatic-differentiation) to learn more about this approach as well as [MLP from scratch](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/).

### Backprop the functional way

Here we will implement in `numpy` a different approach mimicking the functional approach of [JAX](https://jax.readthedocs.io/en/latest/index.html) see [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#).

Each function will take 2 arguments: one being the input `x` and the other being the parameters `w`. For each function, we build 2 **vjp** functions taking as argument a gradient $\mathbf{u}$ and corresponding to $J_{\mathbf{f}}(\mathbf{x})$ and $J_{\mathbf{f}}(\mathbf{w})$ so that these functions return $J_{\mathbf{f}}(\mathbf{x})^T \mathbf{u}$ and $J_{\mathbf{f}}(\mathbf{w})^T \mathbf{u}$ respectively. To summarize, for $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{w} \in \mathbb{R}^d$, and, $\mathbf{f}(\mathbf{x},\mathbf{w}) \in \mathbb{R}^m$,
\begin{align*}
{\bf jvp}_\mathbf{x}(\mathbf{u}) &= J_{\mathbf{f}}(\mathbf{x})^T \mathbf{u}, \text{ with } J_{\mathbf{f}}(\mathbf{x})\in\mathbb{R}^{m\times n}, \mathbf{u}\in \mathbb{R}^m\\
{\bf jvp}_\mathbf{w}(\mathbf{u}) &= J_{\mathbf{f}}(\mathbf{w})^T \mathbf{u}, \text{ with } J_{\mathbf{f}}(\mathbf{w})\in\mathbb{R}^{m\times d}, \mathbf{u}\in \mathbb{R}^m
\end{align*}
Then backpropagation is simply done by first computing the gradient of the loss and then composing the **vjp** functions in the right order.

## Practice

- intro to JAX: autodiff the functional way [autodiff\_functional\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_empty.ipynb) and its solution [autodiff\_functional\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_sol.ipynb)
- Linear regression in JAX [linear\_regression\_jax.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb)