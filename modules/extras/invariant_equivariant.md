@def title = "Invariant Equivaraint layers"
@def hasmath = true

# Invariant and equivariant layers with applications to GNN, PointNet and Transformers

__author: [Marc Lelarge](https://www.di.ens.fr/~lelarge/), course: [dataflowr](https://dataflowr.github.io/website/)__

date: April 23, 2021

## Invariant and equivariant functions

As shown in the [module on GNN](https://dataflowr.github.io/website/modules/graph3/), invariant and equivariant functions are crucial for GNN. For example, the message passing GNN (MGNN) layer is defined by:
$$
\label{eq:gnnlayer}h^{\ell+1}_i  = f(h^\ell_i , \{\{ h^\ell_j\}\}_{j\sim i}),
$$
where $i\sim j$ means that nodes $i$ and $j$ are neighbors and the function $f$ should not depend on the order of the elements in the multiset $\{\{ h^\ell_j\}\}_{j\sim i}$. This layer is applied in parallel to all nodes (with the same function $f$) producing a mapping from ${\bf h}^\ell = (h^\ell_1\dots, h^\ell_n)$ to $F({\bf h}^\ell) = {\bf h}^{\ell+1}$ with $F:\mathbb{R}^n \to \mathbb{R}^n$ where $n$ is the number of nodes in the graph (and only real hidden states are considered for simplicity). It is easy to see that $F$ is an equivariant function, i.e. permuting its input will permute its output. 

Another example of invariant and equivariant functions is given by the attention layer $\text{Attention}(Q,K,V) = Z$ defined for $Q$ a tensor of row queries, $K$ the keys and $V$ the values, $Q,K,V\in \mathbb{R}^{n\times d}$ by
$$
Z_j = \sum_{i=1}^n \text{softmax}_i(Q_jK_i^T) V_i.
$$
The queries are obtained from a tensor $X\in \mathbb{R}^{n\times c}$ by $Q= XW_Q^T$ and the keys and values are obtained from a tensor $X' \in \mathbb{R}^{n\times c'}$  by $K = X' W_K^T$ and $V = X' W_V^T$.
We see that when the queries are fixed, the attention layer is invariant in the pair (keys, values):
$$
Z_j = \sum_{i=1}^n \text{softmax}_{i}(Q_j K_{\sigma(i)}^T) V_{\sigma(i)},
$$
hence $\text{Attention}(X,X')$ is invariant in $X'$. Similarly, when the pair (keys, values) is fixed, the attention layer is equivariant in the queries:
$$
Z_{\sigma(j)} = \sum_{i=1}^n \text{softmax}_{i}(Q_{\sigma(j)}K_{i}^T) V_{i},
$$
hence $\text{Attention}(X,X')$ is equivariant in $X$.
If $X'=X$, we get the self-attention layer 
so that $\text{SelfAttention}(X) = \text{Attention}(X,X)$ is equivariant in $X$.

In this post, we will **characterize invariant and equivariant functions** following the ideas given in the paper [Deep Sets](https://arxiv.org/abs/1703.06114).

## Representation of invariant and equivariant functions

We start with some definitions.

For a vector ${\bf x} = (x_1,\dots, x_n)\in \mathbb{R}^n$ and a permutation $\sigma \in \mathcal{S}_n$, we define
$$
\sigma \star {\bf x} = (x_{\sigma^{-1}(1)},\dots, x_{\sigma^{-1}(n)})
$$

**Definitions:**
- A function $f:\mathbb{R}^n\to \mathbb{R}$ is **invariant** if for all ${\bf x}$ and all $\sigma \in \mathcal{S}_n$, we have $f(\sigma \star {\bf x}) = f({\bf x})$.
- A function $f:\mathbb{R}^n\to \mathbb{R}^n$ is **equivariant** if for all ${\bf x}$ and all $\sigma \in \mathcal{S}_n$, we have $f(\sigma \star {\bf x}) = \sigma \star f({\bf x})$.

We can now state our main result:

@@colbox-blue **Theorem**

- **invariant case:** let $f:[0,1]^n \to \R$ be a continuous function. $f$ is invariant if and only if there are continuous functions $\phi: [0,1] \to \R^n$ and $\rho: \R^n\to \R$ such that
$$
\label{eq:inv}f(\bx) = \rho\left( \sum_{i=1}^n \phi(x_i)\right)
$$

- **equivariant case:** let $f:[0,1]^n \to \R^n$ be a continuous function. $f$ is equivariant if and only if there are continuous functions $\phi: [0,1] \to \R^n$ and $\rho: [0,1]\times \R^n\to \R$ such that
$$
\label{eq:equiv}f_j(\bx) = \rho\left( x_j, \sum_{i=1}^n \phi(x_i)\right)
$$
@@

We give some remarks before providing the proof below. For the sake of simplicity, we consider here a fixed number of points $n$ on the unit interval $[0,1]$. For results with a varying number of points, see [On the Limitations of Representing Functions on Sets](https://arxiv.org/abs/1901.09006) and for points in higher dimension $[0,1]^d$ with $d>1$, see [On Universal Equivariant Set Networks](https://arxiv.org/abs/1910.02421) and [Expressive Power of Invariant and Equivariant Graph Neural Networks](https://arxiv.org/abs/2006.15646).

Our proof will make the mapping $\phi$ explicit and it will not depend on the function $f$. The mapping $\phi$ can be seen as an embedding of the points in $[0,1]$ in a space of high-dimension. Indeed this embedding space has to be of dimension at least the number of points $n$ in order to ensure universality. This is an important remark as in learning scenario, the size of the embedding is typically fixed and hence will limit the expressiveness of the algorithm.

Coming back to the GNN layer \eqref{eq:gnnlayer}, our result on the invariant case tells us that we can always rewrite it as:
$$
\label{eq:gnnlayer2}h^{\ell+1}_i  =\rho\left( h_i^{\ell}, \sum_{j\sim i} \phi(h^\ell_j)\right),
$$
and the dimension of the embedding $\phi(h)$ needs to be of the same order as the maximum degree in the graph. Note that \eqref{eq:gnnlayer2} is not of the form of \eqref{eq:equiv} as the sum inside the $\rho$ function is taken only on neighbors. Indeed, we know that message passing GNN are not universal (see [Expressive Power of Invariant and Equivariant Graph Neural Networks](https://arxiv.org/abs/2006.15646)).

As a last remark, note that the original [PointNet](https://arxiv.org/abs/1612.00593) architecture $f$ is of the form $f_i(\bx) = \rho(x_i)$ which is not universal equivariant. Indeed, it is impossible to approximate the equivariant function $g_i(\bx) = \sum_i x_i$ as shown below (we denote $\be_1=(1,0,\dots,0)$):
$$
\|f(0) - g(0)\|^2 = n \rho(0)^2\\
\|f(\be_1) -g(\be_1)\|^2 = (\rho(1)-1)^2 + (n-1)(\rho(0)-1)^2\geq (n-1)(\rho(0)-1)^2,
$$
and these quantities cannot be small together. Hence PointNet is not universal equivariant but as shown in [On Universal Equivariant Set Networks](https://arxiv.org/abs/1910.02421), modifying PointNet by adding the term $ \sum_{i=1}^n \phi(x_i)$ inside the $\rho$ function as in \eqref{eq:equiv} makes it universal equivariant. We refer to [Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/abs/1912.10077) for similar results about transformers based on self-attention.


## Proof of the Theorem

We first show that the equivariant case is not more difficult than the invariant case. Assume that we proved the invariant case. Consider a permutation $\sigma\in \Sc_n$ such that $\sigma(1)=1$ so that $f(\sigma \star {\bf x}) = \sigma \star f({\bf x})$ gives for the first component:
$$
f_1(x_1,x_{\sigma(2)},\dots, x_{\sigma(n)}) = f_1(x_1,x_2,\dots, x_n).
$$
For any $x_1$, the mapping $(x_2,\dots, x_n) \mapsto f_1(x_1, x_2,\dots, x_n)$ is invariant. Hence by \eqref{eq:inv}, we have
$$
f_1(x_1,x_2,\dots, x_n) = \rho\left(x_1, \sum_{i\neq 1}\phi(x_i) \right)
$$
Now consider a permutation such that $\sigma(1)=k, \sigma(k)=1$ and $\sigma(i)=i$ for $i\neq 1,k$, then we have
\begin{equation}
f_k(x_1,x_2,\dots, x_n) = f_1(x_k,x_2\dots, x_1,\dots x_n),
\end{equation}
hence $f_k(x_1,x_2,\dots, x_n)=\rho\left(x_k, \sum_{i\neq k}\phi(x_i) \right)$ and \eqref{eq:equiv} follows.

Hence, we only need to prove \eqref{eq:inv} and follow the proof given in [Deep Sets](https://arxiv.org/abs/1703.06114). We start with a crucial result stating that a set of $n$ real points is characterized by the first $n$ moments of its empirical measure. Let see what it means for $n=2$: we can recover the values of $x_1$ and $x_2$ from the quantities $p_1=x_1+x_2$ and $p_2=x_1^2+x_2^2$. To see that this is correct, note that
$$
p_1^2 = x_1^2+2x_1x_2+x_2^2 = p_2+2x_1x_2,
$$
so that $x_1x_2 = \frac{p_1^2-p_2}{2}$. As a result, we have
$$
(x-x_1)(x-x_2) = x^2-p_1x+\frac{p_1^2-p_2}{2},
$$
and clearly $x_1$ and $x_2$ can be recovered as the roots of this polynomial whose coefficients are functions of $p_1$ and $p_2$. The result below extends this argument for a general $n$:
@@colbox-blue **Proposition**

Let $\Phi:[0,1]_{\leq}^n \to \mathbb{R}^{n}$, where $[0,1]_{\leq}^n = \{ \bx\in [0,1]^n,\: x_1\leq x_2\leq \dots\leq x_n\}$, be defined by
$$
\Phi(x_1,x_2,\dots, x_n) = \left( \sum_i x_1, \sum_i x_i^2,\dots, \sum_i x_i^n\right)
$$
is injective and has a continuous inverse mapping.@@

The proof follows from [Newton's identities](https://en.wikipedia.org/wiki/Newton%27s_identities). For $k\leq n$, we denote by $p_k = \sum_{i=1}^n x_i^k$ the power sums and by $e_k$ the [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial) (note that all polynomials are function of the $x_1,\dots, x_n$):
\begin{equation}
e_0 = 1\\
e_1 = \sum_i x_i\\
e_2 = \sum_{i < j} x_i x_j\\ \dots
\end{equation}
From Newton's identities, we have for $k\leq n$,
$$
k e_k = \sum_{i=1}^k (-1)^{i-1}e_{k-i}p_i,
$$
so that, we can express the elementary symmetric polynomials from the power sums:
\begin{equation}
e_1 = p_1\\
2e_2 = e_1p_1-p_2=p_1^2-p_2\\
3e_3 = e_2p_2-e_1p_2+p_3 = \frac{1}{2}p_1^3-\frac{3}{2}p_1p_2+p3\\
\dots
\end{equation}
Note that $\Phi(x_1,x_2,\dots, x_n) = (p_1,\dots, p_n)$ and since
$$
\prod_{i=1}^n (x-x_i) = x^n -e_1x^{n-1}+e_2x^{n-2}\dots + (-1)^n e_n,
$$
if $\Phi(\bx) = \Phi(\by)$ then $\prod_{i=1}^n (x-x_i)=\prod_{i=1}^n (x-y_i)$ so that $\{\{x_1,\dots, x_n\}\} = \{\{y_1,\dots, y_n\}\}$ and $\bx=\by \in [0,1]^n_{\leq}$, showing that $\Phi$ is injective.

Hence we proved that $\Phi:[0,1]^n_{\leq} \to \text{Im}(\Phi)$ where $\text{Im}(\Phi)$ is the image of $\Phi$, is a bijection. We need now to prove that $\Phi^{-1}$ is continuous and we'll prove it directly. Let $\by_k \to \by \in\text{Im}(\Phi)$, we need to show that $\Phi^{-1}(\by_k) \to \Phi^{-1}(\by)$. Now if $\Phi^{-1}(\by_k) \not\to \Phi^{-1}(\by)$, since $[0,1]^M_{\leq}$ is compact, this means that there exists a convergent subsequence of $\Phi^{-1}(\by_{k})$ with $\Phi^{-1}(\by_{m_k}) \to \bx\neq \Phi^{-1}(\by) $. But by continuity of $\Phi$, we have $\by_{m_k} \to \Phi(\bx) = \by$, so that we get a contradiction and hence proved the continuity of $\Phi^{-1}$, finishing the proof of the proposition.

We are now ready to prove \eqref{eq:inv}. Let $\phi:[0,1] \to \R^n$ be defined by $\phi(x) = (x,x^2,\dots, x^n)$ and $\rho = f\circ \Phi^{-1}$. Note that $\rho: \text{Im}(\Phi) \to \R$ and $\sum_{i}\phi(x_i) = \Phi(\bx_{\leq})$, where $\bx_{\leq}$ is the vector $\bx$ with components sorted in non-decreasing order. Hence as soon as f is invariant, we have $f(\bx) = f(\bx_{\leq})$ so that \eqref{eq:inv} is valid. We only need to extend the function $\rho$ from the domain $\text{Im}(\Phi)$ to $\R^n$ in a continuous way. This can be done by considering the projection $\pi$ on the compact $\text{Im}(\Phi)$ and define $\rho(\bx) = f\circ \Phi^{-1}(\pi(\bx))$.

Follow on [twitter](https://twitter.com/marc_lelarge)!

## Thanks for reading!