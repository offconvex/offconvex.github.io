---
layout: post
title: Understanding implicit regularization in deep learning by analyzing trajectories of gradient descent
date: 2019-06-xx 12:00:00
author: Nadav Cohen and Wei Hu
visible: True
---

In his [recent blog post](http://www.offconvex.org/2019/06/03/trajectories/), Sanjeev argued that the conventional view of optimization may be insufficient for understanding deep learning, as the *trajectories* taken by optimizers seem to have profound impact on the quality, i.e. the generalization, of obtained solutions.
As an example of analyzing optimizer trajectories for understanding generalization, he briefly described our [new paper with Yuping Luo](https://arxiv.org/abs/1905.13655), which focuses on deep linear neural networks for the task of [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion).
In this post we'll provide more details on the trajectory analysis carried out in the paper, and explain why we believe it is necessary for comprehending the *implicit regularization* at play.

Before we begin, let us recall the setting of matrix completion.
We are given a collection of entries $\\\{ M_{i, j} \\\}_{(i, j) \in \Omega}$ from some unknown *ground truth* matrix $M$, and our goal is to recover its remaining entries.
This can be thought of as a classification (regression) problem, where the training examples are the observed entries of $M$, the model is a matrix $W$ trained with the loss:
$$L(W) = \sum\nolimits_{(i, j) \in \Omega} (W_{i, j} - M_{i, j})^2 ~,$$
and generalization corresponds to how similar $W$ is to $M$ in the unobserved locations.
Obviously the problem is ill-posed if we assume nothing about $M$ $-$ the loss $L(W)$ is underdetermined, i.e. has multiple global minima, and it would be impossible to tell (without access to unobserved entries) if one solution is better than another.
The standard assumption (which has many [practical applications](https://en.wikipedia.org/wiki/Matrix_completion#Applications)) is that the ground truth matrix $M$ is low-rank, and thus the goal is to find, of all global minima for the loss $L(W)$, one which has minimal rank.

## Linear Neural Networks

Linear neural networks are fully-connected neural networks with linear (no) activation. 
Namely, a depth-$N$ linear neural network is a parameterization of a linear mapping as a product of $N$ matrices: $W = W_N W_{N-1} \cdots W_1$.
Besides being more amenable to analysis, a major advantage (from a theoretical perspective) of linear neural networks over non-linear models is that they decouple implicit regularization from expressiveness $-$ the effect of a linear neural network's architecture (depth, hidden layer widths) on the implicit regularization of gradient descent boils down to an implicit algorithm induced on the *end-to-end matrix* $W$.
We have characterized this algorithm, and specifically, the dynamics it entails for the singular value decomposition of $W$.

## Implicit Regularization Towards Low-Rank

Suppose we are interested in learning a matrix $W$ through minimization of some training loss $L(W)$ (e.g. the loss of a matrix completion problem $-$ see above).
Denote the singular values of $W$ by $\\{ \sigma_r \\}_r$, and the corresponding left and right singular vectors by $\\{ \mathbf{u}_r \\}_r$ and $\\{ \mathbf{v}_r \\}_r$ respectively.
One can show that minimizing $L(W)$ via gradient descent with small learning rate $\eta$ leads the singular values of $W$ to evolve by:
$$ \sigma_r(t + 1) \leftarrow \sigma_r(t) - \eta \cdot \langle \nabla L(W(t)) , \mathbf{u}_r(t) \mathbf{v}_r^\top(t) \rangle ~,
\qquad (1)$$
which means that the movement of a singular value is proportional to the projection of the gradient on the corresponding singular component.

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

## Are Trajectories Really Necessary?

As Sanjeev articulated in [his post](http://www.offconvex.org/2019/06/03/trajectories/), the fact that implicit regularization is key to generalization in deep learning (see [Behnam Neyshabur's PhD thesis](https://arxiv.org/pdf/1709.01953.pdf) if need convincing) suggests that the conventional view of optimization is insufficient for understanding the effects at play.
Nonetheless, one could still hope to be able to capture the implicit regularization through the language of standard regularizers, thereby circumventing the need for tedious reasoning about trajectories.
For example, it is known that over linear models, i.e. depth-$1$ networks, gradient descent finds the solution with minimal Frobenius norm (cf. Section 5 in [Zhang et al.](https://openreview.net/pdf?id=Sy8gdB9xx)), and a common hypothesis is that this persists over more elaborate neural networks, with Frobenius norm potentially replaced by some other norm (or quasi-norm) that depends on network architecture.
[Gunasekar et al.](https://papers.nips.cc/paper/7195-implicit-regularization-in-matrix-factorization.pdf) explicitly conjectured:

> **Conjecture (by Gunasekar et al., informal):**
> Gradient descent (with small learning rate and near-zero initialization) training a depth-$2$ linear neural network for matrix completion finds a solution with minimum [nuclear norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms).

It is known that for matrix completion, if the ground truth is low-rank, certain technical assumptions (e.g. "incoherence") are satisfied and sufficiently many entries are observed, minimizing nuclear norm while fitting all observed entries $-$ a convex program $-$ provides perfect reconstruction (cf. [Candes and Recht](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf)).
Parameterizing the solution as a depth-$2$ linear neural network is traditionally viewed as an alternative approach, named *matrix factorization*.
The conjecture of Gunasekar et al. essentially states that matrix factorization trained by gradient descent implicitly realizes the classical and oftentimes optimal method of nuclear norm minimization.
Besides its elegance, Gunasekar et al. motivated the conjecture with empirical evidence they provided, as well as a proof they gave for a certain restricted setting.

Our trajectory analysis and accompanying experiments (described above) suggest that deepening a linear neural network leads gradient descent to produce solutions closer to low-rank.
One may accordingly hypothesize that adding layers to a matrix factorization, resulting in what we call a *deep matrix factorization*, boils down to changing the implicit regularization from nuclear norm to a different norm (or quasi-norm) that approximates rank more closely.
For example, a natural candidate would be a [Schatten-$p$ quasi-norm](https://en.wikipedia.org/wiki/Schatten_norm) with some $0 < p < 1$.
Surprisingly, we establish that in the restricted setting for which Gunasekar et al. proved their conjecture, nuclear norm is minimized not only by matrix factorizations of depth-$2$, but of arbitrary depth.
This result is shown to disqualify Schatten quasi-norms as the implicit regularization.
Instead, when interpreted through a lens similar to that of Gunasekar et al., it brings forth a conjecture by which the implicit regularization is captured by nuclear norm for any depth.
Since our theory and experiments suggest that depth changes (enhances) the implicit regularization, we are led to question the theoretical direction proposed by Gunasekar et al., and conduct additional experiments to evaluate their conjecture.

As we already mentioned, (under mild technical assumptions) nuclear norm minimization perfectly solves matrix completion if the number of observed entries is sufficiently large.
This means that when observations are abundant, any implicit regularization that leads to exact recovery is indistinguishable from nuclear norm minimization (note that the recovery guarantee that was derived in [Li et al.](http://proceedings.mlr.press/v75/li18a/li18a.pdf) is restricted to such settings, thus its support for the conjecture of Gunasekar et al. is limited). 
The regime most interesting to evaluate is therefore that in which the number of observations is too small for exact recovery by minimization of nuclear norm; here there is room for a different implicit regularization to provide higher quality solutions.
Our empirical evaluations show that in this data-poor regime $-$ precisely where implicit regularization matters most $-$ matrix factorizations consistently outperform nuclear norm minimization, suggesting that they admit a stronger implicit bias towards low-rank, in contrast to the conjecture of Gunasekar et al.
$$\color{red}{\text{TODO: add figure showing reconstruction errors of matrix factorizations vs nuclear norm}}$$

Together, our theory and experiments suggest that neither nuclear norm nor Schatten quasi-norms fully encompass the implicit regularization of linear neural networks for matrix completion.
Adding to that the fact that gradient descent on such models leads the end-to-end matrix to follow dynamics that cannot be emulated via any regularizer (see [Nadav's earlier blog post](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/)), we are led to believe that capturing with a single mathematical norm (or quasi-norm) the implicit regularization of linear neural networks, let alone more complex deep learning models, may not be possible, and a detailed account for optimization trajectories might be necessary.

## Conclusion
