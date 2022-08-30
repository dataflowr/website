[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)

You are viewing the static version of the notebook, you can get the [code (GitHub)](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb) or run it in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)

You can also do the [quizzes](https://dataflowr.github.io/quiz/module2a.html)

# Module 2: PyTorch tensors and automatic differentiation

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=103)

```python
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import numpy as np
```

```python
torch.__version__
```

Tensors are used to encode the signal to process, but also the internal states and parameters of models.

**Manipulating data through this constrained structure allows to use CPUs and GPUs at peak performance.**

Construct a 3x5 matrix, uninitialized:

```python
x = torch.empty(3,5)
print(x.dtype)
print(x)
```

If you got an error this [stackoverflow link](https://stackoverflow.com/questions/50617917/overflow-when-unpacking-long-pytorch) might be useful...

```python
x = torch.randn(3,5)
print(x)
```

```python
print(x.size())
```

torch.Size is in fact a [tuple](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences), so it supports the same operations.

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=272)

```python
x.size()[1]
```

```python
x.size() == (3,5)
```

### Bridge to numpy

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=325)

```python
y = x.numpy()
print(y)
```

```python
a = np.ones(5)
b = torch.from_numpy(a)
print(a.dtype)
print(b)
```

```python
c = b.long()
print(c.dtype, c)
print(b.dtype, b)
```

```python
xr = torch.randn(3, 5)
print(xr.dtype, xr)
```

```python
resb = xr + b
resb
```

```python
resc = xr + c
resc
```

Be careful with types!

```python
resb == resc
```

```python
torch.set_printoptions(precision=10)
```

```python
resb[0,1]
```

```python
resc[0,1]
```

```python
resc[0,1].dtype
```

```python
xr[0,1]
```

```python
torch.set_printoptions(precision=4)
```

### [Broadcasting](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=670)

Broadcasting automagically expands dimensions by replicating coefficients, when it is necessary to perform operations.

1. If one of the tensors has fewer dimensions than the other, it is reshaped by adding as many dimensions of size 1 as necessary in the front; then
2. for every mismatch, if one of the two tensor is of size one, it is expanded along this axis by replicating  coefficients.

If there is a tensor size mismatch for one of the dimension and neither of them is one, the operation fails.

```python
A = torch.tensor([[1.], [2.], [3.], [4.]])
print(A.size())
B = torch.tensor([[5., -5., 5., -5., 5.]])
print(B.size())
C = A + B
```

```python
C
```

The original (column-)vector
\begin{eqnarray}
A = \left( \begin{array}{c}
1\\
2\\
3\\
4\\
\end{array}\right)
\end{eqnarray}
is transformed into the matrix 
\begin{eqnarray}
A = \left( \begin{array}{ccccc}
1&1&1&1&1\\
2&2&2&2&2\\
3&3&3&3&3\\
4&4&4&4&4
\end{array}\right)
\end{eqnarray}
and the original (row-)vector
\begin{eqnarray}
C = (5,-5,5,-5,5)
\end{eqnarray}
is transformed into the matrix
\begin{eqnarray}
C = \left( \begin{array}{ccccc}
5&-5&5&-5&5\\
5&-5&5&-5&5\\
5&-5&5&-5&5\\
5&-5&5&-5&5
\end{array}\right)
\end{eqnarray}
so that summing these matrices gives:
\begin{eqnarray}
A+C = \left( \begin{array}{ccccc}
6&-4&6&-4&6\\
7&-3&7&-3&7\\
8&-2&8&-2&8\\
9&-1&9&-1&9
\end{array}\right)
\end{eqnarray}

### In-place modification

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=875)

```python
x
```

```python
xr
```

```python
print(x+xr)
```

```python
x.add_(xr)
print(x)
```

Any operation that mutates a tensor in-place is post-fixed with an `_`

For example: `x.fill_(y)`, `x.t_()`, will change `x`.

```python
print(x.t())
```

```python
x.t_()
print(x)
```

### Shared memory

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=990)

Also be careful, changing the torch tensor modify the numpy array and vice-versa...

This is explained in the PyTorch documentation [here](https://pytorch.org/docs/stable/torch.html#torch.from_numpy):
The returned tensor by `torch.from_numpy` and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. 

```python
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
```

```python
a[2] = 0
print(b)
```

```python
b[3] = 5
print(a)
```

### Cuda

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=1120)

```python
torch.cuda.is_available()
```

```python
#device = torch.device('cpu')
device = torch.device('cuda') # Uncomment this to run on GPU
```

```python
x.device
```

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z,z.type())
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

```python
x = torch.randn(1)
x = x.to(device)
```

```python
x.device
```

```python
# the following line is only useful if CUDA is available
x = x.data
print(x)
print(x.item())
print(x.cpu().numpy())
```

# Simple interfaces to standard image data-bases

[Video timestamp](https://youtu.be/BmAS8IH7n3c?t=1354)

An example, the [CIFAR10](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.CIFAR10) dataset.

```python
import torchvision

data_dir = 'content/data'

cifar = torchvision.datasets.CIFAR10(data_dir, train = True, download = True)
cifar.data.shape
```

Documentation about the [`permute`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute) operation.

```python
x = torch.from_numpy(cifar.data).permute(0,3,1,2).float()
x = x / 255
print(x.type(), x.size(), x.min().item(), x.max().item())
```

Documentation about the [`narrow(input, dim, start, length)`](https://pytorch.org/docs/stable/torch.html#torch.narrow) operation.

```python
# Narrows to the first images, converts to float
x = torch.narrow(x, 0, 0, 48)
```

```python
x.shape
```

```python
# Showing images
def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(20,10))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
show(torchvision.utils.make_grid(x, nrow = 12))
```

```python
# Kills the green and blue channels
x.narrow(1, 1, 2).fill_(0)
show(torchvision.utils.make_grid(x, nrow = 12))
```

# Autograd: automatic differentiation

[Video timestamp](https://youtu.be/Z6H3zakmn6E?t=40)

When executing tensor operations, PyTorch can automatically construct on-the-fly the graph of operations to compute the gradient of any quantity with respect to any tensor involved.

To be more concrete, we introduce the following example: we consider parameters $w\in \mathbb{R}$ and $b\in \mathbb{R}$ with the corresponding function:
\begin{eqnarray*}
\ell = \left(\exp(wx+b) - y^* \right)^2
\end{eqnarray*}

Our goal here, will be to compute the following partial derivatives:
\begin{eqnarray*}
\frac{\partial \ell}{\partial w}\mbox{ and, }\frac{\partial \ell}{\partial b}.
\end{eqnarray*}

The reason for doing this will be clear when you will solve the practicals for this lesson!

You can decompose this function as a composition of basic operations. This is call the forward pass on the graph of operations.
![backprop1](https://dataflowr.github.io/notebooks/Module2/img/backprop1.png)

Let say we start with our model in `numpy`:

```python
w = np.array([0.5])
b = np.array([2])
xx = np.array([0.5])#np.arange(0,1.5,.5)
```

transform these into `tensor`:

```python
xx_t = torch.from_numpy(xx)
w_t = torch.from_numpy(w)
b_t = torch.from_numpy(b)
```

[Video timestamp](https://youtu.be/Z6H3zakmn6E?t=224)

A `tensor` has a Boolean field `requires_grad`, set to `False` by default, which states if PyTorch should build the graph of operations so that gradients with respect to it can be computed.

```python
w_t.requires_grad
```

We want to take derivative with respect to $w$ so we change this value:

```python
w_t.requires_grad_(True)
```

We want to do the same thing for $b$ but the following line will produce an error!

```python
b_t.requires_grad_(True)
```

Reading the error message should allow you to correct the mistake!

```python
dtype = torch.float64
```

```python
b_t = b_t.type(dtype)
```

```python
b_t.requires_grad_(True)
```

[Video timestamp](https://youtu.be/Z6H3zakmn6E?t=404)

We now compute the function:

```python
def fun(x,ystar):
    y = torch.exp(w_t*x+b_t)
    print(y)
    return torch.sum((y-ystar)**2)

ystar_t = torch.randn_like(xx_t)
l_t = fun(xx_t,ystar_t)
```

```python
l_t
```

```python
l_t.requires_grad
```

After the computation is finished, i.e. *forward pass*, you can call `.backward()` and have all the gradients computed automatically.

```python
print(w_t.grad)
```

```python
l_t.backward()
```

```python
print(w_t.grad)
print(b_t.grad)
```

[Video timestamp](https://youtu.be/Z6H3zakmn6E?t=545)

Let's try to understand these numbers...

![backprop2](https://dataflowr.github.io/notebooks/Module2/img/backprop2.png)

```python
yy_t = torch.exp(w_t*xx_t+b_t)
print(torch.sum(2*(yy_t-ystar_t)*yy_t*xx_t))
print(torch.sum(2*(yy_t-ystar_t)*yy_t))
```

`tensor.backward()` accumulates the gradients in  the `grad` fields  of tensors.

```python
l_t = fun(xx_t,ystar_t)
l_t.backward()
```

```python
print(w_t.grad)
print(b_t.grad)
```

By default, `backward` deletes the computational graph when it is used so that you will get an error below:

```python
l_t.backward()
```

```python
# Manually zero the gradients
w_t.grad.data.zero_()
b_t.grad.data.zero_()
l_t = fun(xx_t,ystar_t)
l_t.backward(retain_graph=True)
l_t.backward()
print(w_t.grad)
print(b_t.grad)
```

The gradients must be set to zero manually. Otherwise they will cumulate across several _.backward()_ calls. 
This accumulating behavior is desirable in particular to compute the gradient of a loss summed over several “mini-batches,” or the gradient of a sum of losses.

[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)
