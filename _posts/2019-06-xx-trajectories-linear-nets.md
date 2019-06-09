---
layout: post
title: Understanding implicit regularization in deep learning by analyzing trajectories of gradient descent
date: 2019-06-xx 12:00:00
author: Nadav Cohen and Wei Hu
visible: True
---

Last week Sanjeev argued that the conventional view of optimization may be insufficient for understanding deep learning, as the particular trajectories taken by the algorithm seem to have profound impact on the quality (generalization) of the obtained solution.
As an illustrative example, he briefly described our new paper with Yuping Luo, which studies matrix completion by deep linear neural networks.
In this blog post we provide details on the trajectory analysis carried out in this paper, and why we believe it is necessary for understanding the implicit regularization at play.

## Linear Neural Networks

Linear neural networks are fully-connected neural networks with linear (no) activation. 
Specifically, a depth N linear neural network is a parameterization of a linear mapping as a product of N matrices: W = W_N W_{N-1} \cdots W_1.
Besides being more amenable to theoretical analysis, a major advantage of linear neural networks over non-linear models is that they decouple implicit regularization from expressiveness.
That is, the effect of a linear neural network's architecture (hidden dimensions, depth) on the implicit regularization of gradient descent boils down to an algorithm induced on the end-to-end mapping W.
We have characterized this algorithm, and specifically the dynamics it entails for the singular value decomposition of W.

## Implicit Regularization Towards Low-Rank

Let L(W) be some training loss over a matrix W.
Denote the singular values of W by {sigma_r}_r, and the corresponding left and right singular vectors by {u_r}_r and {v_r}_r respectively.
One can show that running gradient descent on L(W) with small learning rate leads the singular values of W to evolve by:
[equation]
In words, the movement of a singular value is equal to the projection of the gradient on the corresponding singular component.
Now suppose that we parameterize W with a linear neural network, i.e. via W = W_N W_{N-1} \cdots W_1.
In previous work (described in Nadav's earlier blog post) we have shown that running gradient descent over the linear neural network, with small learning rate and initialization close to the origin, leads the end-to-end matrix W to follow the update rule:
[equation]
What we show in the new paper is that this implies the following dynamics for the singular values:
[equation]
Comparing this equation to the one above, we see that overparameterizing the loss L(W) with a depth-N linear neural network introduces the multiplicative factors N(\sigma_r^2)^{1-1/N} to the evolution of singular values.
While the constant N does not differentiate between singular values (can be absorbed into the learning rate), the terms (\sigma_r^2)^{1-1/N} do --- they enhance movement of large singular values, and on the hand attenuate that of small ones.
Moreover, the enhancement/attenuation becomes more significant as N (depth of the factorization) grows.
[figure]
The enhancement/attenuation effect described above leads a singular value to progress very slowly after initialization, when close to zero, and then, upon reaching a certain threshold, movement rapidly, with the transition from slow to rapid movement being sharper with a deeper network (larger N).
When the loss L(W) is underdetermined, i.e. has multiple global minima, these dynamics promote solutions that have a few large singular values and many small ones, with a gap that is more extreme the deeper the network is. 
This is an implicit regularization towards low-rank, which intensifies with depth.
We provide theoretical illustrations and empirical results verifying the intuition, i.e. showing that gradient descent (with small learning rate and initialization near the origin) training a linear neural network over an underdetermined loss indeed has a tendency towards low-rank solutions, which intensifies with depth.
For example, the following plots show the evolution of singular values throughout gradient descent optimization for matrix completion --- as can be seen, depth leads to solutions with a spectrum closer to low-rank, thereby improving reconstruction.
[figure]

## Are Trajectories Really Necessary?

Since the work of Zhang et al., there is little dispute over the fact that implicit regularization is key to the success of deep learning.
This suggests that adopting a conventional view of optimization (see Sanjeev's post) is insufficient for understanding the effects at play.
However, one can still hope to be able to capture the implicit regularization through the language of standard regularizers, thereby avoiding the undertaking of analyzing trajectories.
For example, it is known that over linear models (depth-1 network), when the loss is underdetermined, gradient descent finds the solution with minimal Frobenius norm, and a widespread belief is that this persists with more elaborate neural networks, with Frobenius norm being replaced by some other norm (or quasi-norm).
Gunasekar et al. formally conjectured that for matrix completion with a depth-2 linear neural network, a.k.a. matrix factorization, gradient descent (with small learning rate and initialization near the origin) implicitly minimizes the nuclear norm --- a convex relaxation of rank.
Their conjecture was motivated by empirical evidence they provided, as well as a proof they gave for a certain restricted setting.

Given our finding by which deeper linear neural networks lead gradient descent to provide lower rank solutions, one may hypothesize that adding depth to a matrix factorization --- resulting in what we call deep matrix factorization --- boils down to changing the implicit regularization from nuclear norm to a different norm (or quasi-norm) that approximates rank more closely.
For example, a natural candidate would be a Schatten-p quasi-norm with some 0 < p < 1.
Surprisingly, we prove that in the restricted setting for which Gunasekar et al. proved their conjecture, nuclear norm is minimized by matrix factorizations not only of depth-2, but of arbitrary depth.
This disqualifies Schatten quasi-norms as the implicit regularization.
Instead, when interpreted through the lens of Gunasekar et al., it brings forth a conjecture by which the implicit regularization in matrix factorization is captured by nuclear norm for any depth.
Since our theory and experiments suggest that depth changes (enhances) the implicit regularization, we are led to question the nuclear norm conjecture, and conduct additional experiments to evaluate its validity.

It is known from the matrix completion literature (see for example the classic paper by Candes and Recht) that (under mild technical assumptions) when the number of observed entries is sufficiently large, nuclear norm minimization yields exact recovery.
This means that in such scenarios, it is impossible to distinguish between nuclear norm minimization and a different implicit regularization that also perfectly recovers (note that the later work of Tengyu et al. supporting the conjecture of Gunasekar et al. was limited to these scenarios).
The regime that is most interesting to evaluate is therefore when the number of observed entries is too small for exact recovery by nuclear norm minimization --- here there is room for different implicit regularizations to provide higher quality solutions.
Our empirical results show that in this regime, matrix factorizations consistently outperform nuclear norm minimization, suggesting that their implicit regularization admits stronger bias towards low-rank, in contrast to the conjecture of Gunasekar et al.
[figure]

Together, our theory and experiments suggest that neither nuclear norm nor Schatten quasi-norms fully capture the implicit regularization of linear neural networks for matrix completion.
Adding to that the fact that gradient descent on such models leads the end-to-end matrix to follow dynamics that cannot be emulated via any regularizer (see Nadav's earlier blog post), we are led to believe that capturing by a single mathematical norm (or quasi-norm) the implicit regularization of linear neural networks, let alone more complex deep learning models, may not be possible, and a detailed account for the optimization trajectories might be necessary.

## Conclusion
