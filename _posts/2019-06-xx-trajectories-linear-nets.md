---
layout: post
title: Understanding implicit regularization in deep learning by analyzing trajectories of gradient descent
date: 2019-06-xx 12:00:00
author: Nadav Cohen and Wei Hu
visible: False
---

Sanjeev's [recent blog post](http://www.offconvex.org/2019/06/03/trajectories/) suggested that the conventional view of optimization may be insufficient for fully understanding deep learning, since value of the training objective seems insufficient to understand *generalization*, and one needs to consider the *trajectory*  of optimization. 
One of the illustrative examples  was our [new paper with Yuping Luo](https://arxiv.org/abs/1905.13655), which studies how to use deep linear neural networks for solving [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) better than the classic convex programming approach. 
This post provides more details on this result.

Recall that in *matrix completion* 
we are given some entries $\\\{ M_{i, j} \\\}_{(i, j) \in \Omega}$ of an unknown *ground truth* matrix $M$, and our goal is to recover the remaining entries.
This can be thought of as a classification (regression) problem, where the training examples are the observed entries of $M$, the model is a matrix $W$ trained with the loss:
$$L(W) = \sum\nolimits_{(i, j) \in \Omega} (W_{i, j} - M_{i, j})^2 ~,$$
and generalization corresponds to how similar $W$ is to $M$ in the unobserved locations.
Obviously the problem is ill-posed if we assume nothing about $M$ $-$ the loss $L(W)$ is underdetermined, i.e. has multiple global minima, and it would be impossible to tell (without access to unobserved entries) if one solution is better than another.
The standard assumption (which has many [practical applications](https://en.wikipedia.org/wiki/Matrix_completion#Applications)) is that the ground truth matrix $M$ is low-rank, and thus the goal is to find, from among all global minima for the loss $L(W)$, one with minimal rank. The classic algorithm finds the matrix of minimum [nuclear norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms). that also fits all the observed entries. Under some reasonable assumptions and given enough matrix entries, this algorithm recovers the ground truth matrix exactly(cf. [Candes and Recht](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf)). We're interested in the regime when the number of revealed entries is too small for the classic algorithm to succeed, and it can be beaten by a simple deep learning approach described next. 

## Linear Neural Networks

A depth $N$ linear neural network is a fully-connected neural network with linear  activation (i.e., no nonlinearity). If $W_i$ is the matrix in layer $i$ the end-to-end function is given by $W = W_N W_{N-1} \cdots W_1$. Our method to solve matrix completion involves finding the unknown matrix as such a $W$, where depth $N\geq 3$. This can be viewed as a deep learning problem with squared loss, and gradient descent can be implemented via backpropagation as usual. Note that the loss does not have any regularizer term involving the layer matrices $W_i$'s or their norms. 

At first sight, such an algorithm seems dangerously naive, since a multilayer linear net is equivalent to a single-layer net. However, matrix completion is an underdetermined problem with multiple optima. Running simple gradient descent (GD) makes us choose one of these optima. Thus this setup isolates the role of GD in selecting optima that generalize well. 

Note that the subcase $N=2$ is also a traditional approach to matrix completion,  named *matrix factorization*. By analogy, we can refer to the case $N\geq 3$ as *deep matrix factorization*. The figure below shows that $N=3$ solves matrix completion better than $N=2$ and the nuclear norm method. 

>GIVE FIGURE HERE!!! INCLUDING depth 2, nuclear norm, and depth 3.

The main contribution of the paper is to develop theoretical understanding of this phenomenon. 

## Implicit Regularization Towards Low-Rank

We are interested in understanding what end-to-end matrix $W$ emerges when we use gradient descent (GD) to minimize a convex loss $L(W)$. (The specific loss $\L(\cdot)$ used for matrix completion was the squared loss. Note that it is convex in $W$ but not in the $W_i$'s.)  Denote the singular values of $W$ by $\\{ \sigma_r \\}_r$, and the corresponding left and right singular vectors by $\\{ \mathbf{u}_r \\}_r$ and $\\{ \mathbf{v}_r \\}_r$ respectively. 

> Minimizing $L(W)$ via GD with small learning rate $\eta$ leads the singular values of $W$ to evolve by:
$$ \sigma_r(t + 1) \leftarrow \sigma_r(t) - \eta \cdot \langle \nabla L(W(t)) , \mathbf{u}_r(t) \mathbf{v}_r^\top(t) \rangle ~,
\qquad (1).$$

The above statement implies that the movement of a singular value is proportional to the projection of the gradient on the corresponding singular component.

Now suppose that we parameterize $W$ with a depth-$N$ linear neural network, i.e. as $W = W_N W_{N-1} \cdots W_1$.
In previous work (described in [Nadav's earlier blog post](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/)) we have shown that running gradient descent on the linear neural network, with small learning rate $\eta$ and initialization close to the origin, leads the end-to-end matrix $W$ to evolve by:
$$ W(t+1) \leftarrow W(t) - \eta \cdot \sum\nolimits_{j=1}^{N} \left[ W(t) W^\top(t) \right]^\frac{j-1}{N} \nabla{L}(W(t)) \left[ W^\top(t) W(t) \right]^\frac{N-j}{N} ~.$$
In the new paper we prove that this implies the following dynamics for the singular values:
$$ \sigma_r(t + 1) \leftarrow \sigma_r(t) - \eta \cdot \color{purple}{N \cdot (\sigma_r(t))^{2 - 2 / N}} \cdot \langle \nabla L(W(t)) , \mathbf{u}_r(t) \mathbf{v}_r^\top(t) \rangle ~. \qquad (2)$$
Comparing this to Equation $(1)$, we see that (over-)parameterizing the loss $L(W)$ with a depth-$N$ linear neural network introduces the multiplicative factors $\color{purple}{N \cdot (\sigma_r(t))^{2 - 2 / N}}$ to the evolution of singular values.
While the constant $N$ does not change relative dynamics (can be absorbed into the learning rate $\eta$), the terms $(\sigma_r(t))^{2 - 2 / N}$ do $-$ they enhance movement of large singular values, and on the hand attenuate that of small ones.
Moreover, the enhancement/attenuation becomes more significant as $N$ (network depth) grows.
$$\color{red}{\text{TODO: add illustrative figure}}$$

The enhancement/attenuation effect induced by a linear neural network ($\color{purple}{\text{purple}}$ term in Equation $(2)$) leads each singular value to progress very slowly after initialization, when close to zero, and then, upon reaching a certain threshold, move rapidly, with the transition from slow to rapid movement being sharper in case of a deeper network (larger $N$).
If the loss $L(W)$ is underdetermined (has multiple global minima) these dynamics promote solutions that have a few large singular values and many small ones (that have yet to reach the phase transition between slow to rapid movement), with a gap that is more extreme the deeper the network is. 
This is an implicit regularization towards low-rank, which intensifies with depth.
In the paper we support the intuition with empirical evaluations and theoretical illustrations demonstrating how adding depth to a linear neural network subject to an underdetermined loss leads gradient descent (with small learning rate and initialization near the origin) to produce solutions closer to low-rank.
For example, the following plots, corresponding to a task of matrix completion, show evolution of singular values throughout training of linear neural networks with varying depths $-$ as can be seen, adding layers admits a final solution whose spectrum is closer to low-rank, thereby improving generalization (reconstruction of low-rank ground truth).
$$\color{red}{\text{TODO: add figure showing evolution of singular values and final reconstruction errors}}$$

## Do Trajectories minimize some regularized objective?

In recent years researchers have come to realize the power of implicit regularization due to the choice of optimization algorithm, which directly motivated us. However, given the strong gravitational pull of the optimization worldview, these papers  focused on trying to capture the effect in language of standard regularizers. 
For example, it is known that over linear models, i.e. depth-$1$ networks, gradient descent finds the solution with minimal Frobenius norm (cf. Section 5 in [Zhang et al.](https://openreview.net/pdf?id=Sy8gdB9xx)), and a common hypothesis is that this persists over more elaborate neural networks, with Frobenius norm potentially replaced by some other norm (or quasi-norm) that depends on network architecture.
[Gunasekar et al.](https://papers.nips.cc/paper/7195-implicit-regularization-in-matrix-factorization.pdf) explicitly conjectured:

> **Conjecture (by Gunasekar et al., informal):**
> Gradient descent (with small learning rate and near-zero initialization) training a depth-$2$ linear neural network for matrix completion finds a solution with minimum [nuclear norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms).

This conjecture essentially states that matrix factorization (i.e. depth $2$ linear net) trained by GD is equivalent to the famous method of nuclear norm minimization.
Gunasekar et al. motivated the conjecture with *empirical* evidence, as well as some *mathematical* evidence in the form of a proof for the conjecture in (very) restricted setting.

Given that deep matrix factorization solves matrix completion even better in practice, it would be natural to extend the above conjecture (as we did too at the start of our investigation) to assert that the implicit regularization from  depths $3$ and higher corresponds to minimizing a different norm (or quasi-norm) of $W$ that approximates rank more closely.
For example, a natural candidate would be a [Schatten-$p$ quasi-norm](https://en.wikipedia.org/wiki/Schatten_norm) with some $0 < p < 1$.

However, in the course of our investigation we came to disbelieve the Gunasekar et al style conjectures. 

> (Mathematical evidence against conjecture) In the same simple settings where  Gunasekar et al. proved their conjecture, nuclear norm is minimized not only by matrix factorizations of depth-$2$, but also by factorizations of arbitrary depth.

Since empirically there is a notable difference in performance of depth $3$ vs depth $2$, this 
 calls into question the theoretical direction suggested by Gunasekar et al..

Furthermore, the empirical evidence of Gunasekar et al. also is somewhat incomplete. Their experiments were in the regime where enough matrix entries are revealed that the  nuclear norm algorithm is optimal. Our experiments are in the regime where the number of matrix entries revealed is a bit lower, and the nuclear norm algorithm is suboptimal. Here both depth $2$ and depth $3$ factorizations do better empirically, which suggests to us that even depth $2$ nets have a stronger implicit bias towards low-rank than nuclear norm, in contrast to the conjecture of Gunasekar et al.
$$\color{red}{\text{TODO: add figure showing reconstruction errors of matrix factorizations vs nuclear norm}}$$

Together, our theory and experiments suggest that neither nuclear norm nor Schatten quasi-norms fully encompass the implicit regularization of linear neural networks for matrix completion.
Adding to that the fact that gradient descent on such models leads the end-to-end matrix to follow dynamics that cannot be emulated via any regularizer (see [Nadav's earlier blog post](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/)), we are led to believe that it should not be possible to capture the implicit regularization of linear neural networks with a single mathematical norm (or quasi-norm). Clearly, capturing the effects in more complex deep learning models may be even harder and a detailed account for optimization trajectories might be necessary.

## Conclusion
