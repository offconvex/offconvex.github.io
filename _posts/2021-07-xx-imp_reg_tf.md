---
layout: post
title: Implicit Regularization in Tensor Factorization: Can Tensor Rank Shed Light on Generalization in Deep Learning?
date: 2021-07-xx xx:00:00
author: Noam Razin, Asaf Maman, Nadav Cohen
visible: False
---

Recent efforts to understand implicit regularization in deep learning have led to theoretical focus on matrix factorization, which can be seen as linear neural networks.
This post is based on our [recent paper](https://arxiv.org/pdf/2102.09972.pdf) (to appear at ICML 2021), where a step towards practical deep learning is taken, through investigation of *tensor factorization* &mdash; a model equivalent to a certain type of non-linear neural networks.
It is well known that [most tensor problems are NP-hard](https://arxiv.org/pdf/0911.1393.pdf), and accordingly, the common sentiment is that working with tensors (in both theory and practice) entails extreme difficulties.
However, by adopting a dynamical systems view, we manage to avoid such difficulties, and establish an implicit regularization towards low *tensor rank*.
Our results suggest that tensor rank may shed light on generalization in deep learning.

## Challenge: finding a right measure of complexity

Overparameterized neural networks are mysteriously able to generalize even when trained without any explicit regularization.
Per conventional wisdom, this generalization stems from an *implicit regularization* &mdash; a tendency of gradient-based optimization to fit training examples with predictors of minimal ‘‘complexity.''
A major challenge in translating this intuition to provable guarantees is that we currently lack measures for predictor complexity that are quantitative (admit generalization bounds), and at the same time, capture the essence of natural data (such as images, audio, and text), in the sense that it can be fit with predictors of low complexity.

<div style="text-align:center;">
<!--<img style="width:500px;padding-bottom:10px;padding-top:5px" src="http://www.offconvex.org/assets/imp_reg_tf_data_complexity.png" />-->
<img style="width:500px;padding-bottom:10px;padding-top:5px" src="https://noamrazin.github.io/files/temp/imp_reg_tf_data_complexity.png" />
<br>
<i><b>Figure 1:</b> 
To explain generalization in deep learning, a complexity <br>
measure must allow the fit of natural data with low complexity. On the <br>
other hand, when fitting data which does not admit generalization, <br>
e.g. random data, the complexity should be high. 
</i>
</div>
<br>

## A common testbed: matrix factorization

In the absence of a clear complexity measure for practical neural networks, existing analyses typically focus on simplistic settings where a notion of complexity is apparent. 
A prominent example for such a setting is *matrix factorization* &mdash; matrix completion via linear neural networks. 
This model was discussed quite extensively in previous posts (see [one](http://www.offconvex.org/2019/06/03/trajectories/) by Sanjeev, [one](http://www.offconvex.org/2019/07/10/trajectories-linear-nets/) by Nadav and Wei and [another one](https://www.offconvex.org/2020/11/27/reg_dl_not_norm/) by Nadav), but for completeness we present it again here.

In *matrix completion* we are given a subset of entries from an unknown matrix $W^* \in \mathbb{R}^{d, d'}$, and our task is to predict the unobserved entries. 
This can be viewed as a supervised learning problem with $2$-dimensional inputs, where the label of the input $( i , j )$ is $[W^*]_{i,j}$.
Under such viewpoint, the observed entries are the training set, and the average reconstruction error over unobserved entries is the test error, quantifying generalization.
A predictor can then be seen as a matrix, and a natural notion of complexity is its *rank*.
Indeed, in many real-world scenarios (a famous example is the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)) one is interested in [low rank matrix recovery from incomplete observations](https://arxiv.org/pdf/1601.06422.pdf).

A ‘‘deep learning approach'' to matrix completion is matrix factorization. 
The idea is to use a linear neural network (fully connected neural network with no non-linearity), and fit observations via gradient descent (GD). 
This amounts to optimizing the following objective:

<div style="text-align:center;">
\[
 \min\nolimits_{W_1 , \ldots , W_L} ~ \sum\nolimits_{(i,j) \in observations} \big( [W_L \cdots W_1 ]_{i, j} - [W^*]_{i,j} \big)^2 ~.
\]
</div>

It is obviously possible to constrain the rank of the produced solution by limiting the shared dimensions of the weight matrices $\\{ W_j \\}_j$. 
However, from an implicit regularization standpoint, the case of interest is where rank is unconstrained and the factorization can express any matrix. 
In this case there is no explicit regularization, and the kind of solution we get is determined implicitly by the parameterization and optimization algorithm.

As it turns out, in practice, matrix factorization with near-zero initialization and small step size tends to accurately recover low rank matrices.
This phenomenon (initially identified in [Gunasekar et al. 2017](https://papers.nips.cc/paper/2017/file/58191d2a914c6dae66371c9dcdc91b41-Paper.pdf)) manifests some kind of implicit regularization, whose mathematical characterization drew a lot of interest.
It was initially conjectured that nuclear norm is implicitly minimized ([Gunasekar et al. 2017](https://papers.nips.cc/paper/2017/file/58191d2a914c6dae66371c9dcdc91b41-Paper.pdf)), but recent evidence points towards implicit rank minimization, which stems from incremental learning dynamics (see [Arora et al. 2019](https://papers.nips.cc/paper/2019/file/c0c783b5fc0d7d808f1d14a6e9c8280d-Paper.pdf); [Razin & Cohen 2020](https://papers.nips.cc/paper/2020/file/f21e255f89e0f258accbe4e984eef486-Paper.pdf); [Li et al. 2021](https://openreview.net/pdf/e29b53584bc9017cb15b9394735cd51b56c32446.pdf)). 
Today, it seems we have a relatively firm understanding of generalization in matrix factorization.
There is a complexity measure for predictors &mdash; matrix rank &mdash; by which implicit regularization strives to lower complexity, and the data itself is (in many real-world scenarios) of low complexity (i.e. can be fit with low complexity). 
Jointly, these two conditions lead to generalization.

## Beyond matrix factorization: tensor factorization

Although matrix factorization is interesting on its own behalf, as a theoretical surrogate for deep learning it is limited.
First, it corresponds to a *linear* neural network, thus misses the crucial aspect of non-linearity. 
Second, viewing matrix completion as a prediction problem, it doesn't capture tasks with more than two input variables.
As we now discuss, both of these limitations are lifted if instead of matrices one considers tensors. 

A tensor can be thought of as a multi-dimensional array.
The number of axes in a tensor is referred to as its *order*.
In the task of *tensor completion*, a subset of entries from an unknown tensor $\mathcal{W}^* \in \mathbb{R}^{d_1, \ldots, d_N}$ are given, and the goal is to predict the unobserved entries. 
Analogously to how matrix completion can be viewed as a prediction problem over two input variables, order-$N$ tensor completion can be seen as a prediction problem over $N$ input variables (each corresponding to a different axis).
In fact, any multi-dimensional prediction task with discrete inputs and scalar output can be formulated as a tensor completion problem.
Consider for example the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), and for simplicity assume that image pixels hold one of two values, i.e. are either black or white. 
The task of predicting labels for the $28$-by-$28$ binary images can be seen as an order-$784$ (one axis for each pixel) tensor completion problem, where all axes are of length $2$ (corresponding to the number of values a pixel can take). 
For further details on how general prediction tasks map to tensor completion problems see [our paper](https://arxiv.org/pdf/2102.09972.pdf).

<div style="text-align:center;">
<!--<img style="width:550px;padding-bottom:10px;padding-top:5px" src="http://www.offconvex.org/assets/pred_prob_to_tensor_comp.png" />-->
<img style="width:550px;padding-bottom:10px;padding-top:5px" src="https://noamrazin.github.io/files/temp/pred_prob_to_tensor_comp.png" />
<br>
<i><b>Figure 2:</b> 
Prediction tasks can be viewed as tensor completion problems. <br>
For example, predicting labels for input images with $3$ pixels, each taking <br>
one of $5$ grayscale values, corresponds to completing a $5 \times 5 \times 5$ tensor.
</i>
</div>
<br>
Similarly to matrices, tensors can be factorized. 
The most basic scheme for factorizing tensors, named CANDECOMP/PARAFAC (CP), parameterizes a tensor as a sum of outer products (for information on this scheme, as well as others, see the [excellent survey](http://www.kolda.net/publication/TensorReview.pdf) of Kolda and Bader).
In [our paper](https://arxiv.org/pdf/2102.09972.pdf) and this post, we use the term *tensor factorization* to refer to solving tensor completion by fitting observations via GD over CP parameterization.
The optimized objective resembles what we had for matrix factorization ($\otimes$ here stands for outer product):

<div style="text-align:center;">
\[
\min\nolimits_{ \{ \mathbf{w}_r^n \}_{r , n} } \sum\nolimits_{ (i_1 , ... , i_N) \in observations } \big( \big[  {\textstyle \sum}_{r = 1}^R \mathbf{w}_r^1 \otimes \cdots \otimes \mathbf{w}_r^N \big]_{i_1 , \ldots , i_N} - [\mathcal{W}^*]_{i_1 , \ldots , i_N} \big)^2 ~.
\]
</div>

The notion of rank naturally extends from matrices to tensors.
The *tensor rank* of a given tensor $\mathcal{W}$ is defined to be the minimal number of components (i.e. of outer product summands) $R$ required for CP parameterization to express it.
Note that for order-$2$ tensors, i.e. for matrices, this indeed coincides with matrix rank.
We can explicitly restrict the tensor rank of solutions found by tensor factorization via limiting the number of components $R$. 
However, since our interest lies on implicit regularization, we consider the case where $R$ is large enough for any tensor to be expressed.

By now you might be wondering what does tensor factorization have to do with deep learning.
Apparently, as Nadav mentioned in an [earlier post](http://www.offconvex.org/2020/11/27/reg_dl_not_norm/), analogously to how matrix factorization is equivalent to matrix completion (two-dimensional prediction) via linear neural network, tensor factorization is equivalent to tensor completion (multi-dimensional prediction) with a certain *non-linear* neural network (for the exact details behind the latter equivalence see [our paper](https://arxiv.org/pdf/2102.09972.pdf)). 
It thus represents a setting one step closer to practical neural networks. 

<div style="text-align:center;">
<!--<img style="width:900px;padding-bottom:10px;padding-top:5px" src="http://www.offconvex.org/assets/mf_lnn_tf_nonlinear.png" />-->
<img style="width:900px;padding-bottom:10px;padding-top:5px" src="https://noamrazin.github.io/files/temp/mf_lnn_tf_nonlinear.png" />
<br>
<i><b>Figure 3:</b> 
While matrix factorization corresponds to a linear neural network, <br>
tensor factorization corresponds to a certain non-linear neural network.
</i>
</div>
<br>
As a final piece in the analogy between matrix and tensor factorization, in a [previous paper](https://arxiv.org/pdf/2005.06398.pdf) (described in an [earlier post](https://www.offconvex.org/2020/11/27/reg_dl_not_norm/)) Noam and Nadav demonstrated empirically that (similarly to the phenomenon described above for matrices) tensor factorization with near-zero initialization and small step size tends to accurately recover low rank tensors.
Our goal in the [current paper](https://arxiv.org/pdf/2102.09972.pdf) was to mathematically explain this finding. 
To avoid the [notorious difficulty of tensor problems](https://arxiv.org/pdf/0911.1393.pdf), we chose to adopt a dynamical systems view, and analyze directly the trajectories induced by gradient descent.

## Dynamical analysis: implicit tensor rank minimization

So what can we say about the implicit regularization in tensor factorization? 
At the core of our analysis is the following dynamical characterization of component norms:

> **Theorem:**
> Running gradient flow (GD with infinitesimal step size) over a tensor factorization with near-zero initialization leads component norms to evolve by:
\[ \frac{d}{dt} || \mathbf{w}_r^1 (t) \otimes \cdots \otimes \mathbf{w}_r^N (t) || \propto \color{brown}{|| \mathbf{w}_r^1 (t) \otimes \cdots \otimes \mathbf{w}_r^N (t) ||^{2 - 2/N}} ~,
\]
> where $\mathbf{w}_r^1 (t), \ldots, \mathbf{w}_r^N (t)$ denote the weight vectors at time $t \geq 0$.

According to the theorem above, component norms evolve at a rate that is proportional to their size exponentiated by $\color{brown}{2 - 2 / N}$ (recall that $N$ is the order of the tensor to complete).
Consequently, they are subject to a momentum-like effect, by which they move slower when small and faster when large. 
This suggests that when initialized near zero, components tend to remain close to the origin, and then, upon passing a critical threshold, quickly grow until convergence. 
Intuitively, these dynamics induce an incremental process in which components are learned one after the other, leading to solutions with a few large components and many small ones, i.e. with (approximately) low tensor rank! 

We empirically verified the incremental learning of components in various settings. 
Here is a representative example from one of our experiments (see [the paper](https://arxiv.org/pdf/2102.09972.pdf) for additional examples):

<div style="text-align:center;">
<!--<img style="width:800px;padding-bottom:15px;padding-top:10px;" src="http://www.offconvex.org/assets/tf_dyn_exps.png" />-->
<img style="width:800px;padding-bottom:15px;padding-top:10px;" src="https://noamrazin.github.io/files/temp/tf_dyn_exps.png" />
<br>
<i><b>Figure 4:</b> 
Dynamics of component norms during GD over tensor factorization. <br>
An incremental learning effect is enhanced as initialization scale decreases, <br>
leading to accurate completion of tensor with low tensor rank .
</i>
</div>
<br>
Building on the dynamical characterization of component norms, we were able to prove that with sufficiently small initialization, tensor factorization (approximately) follows a trajectory of (tensor) rank one tensors for an arbitrary amount of time. 
This leads to:

> **Theorem:**
> If tensor completion has a (tensor) rank one solution, then under certain technical conditions, tensor factorization will reach it.

It is worth mentioning that, in a way, our results extend the incremental rank learning dynamics established for matrix factorization (cf. [Arora et al. 2019](https://papers.nips.cc/paper/2019/file/c0c783b5fc0d7d808f1d14a6e9c8280d-Paper.pdf) and [Li et al. 2021](https://arxiv.org/pdf/2012.09839v1.pdf)) to tensor factorization. 
As typical when transitioning from matrices to tensors, this extension entailed challenges that required use of different techniques.

## Tensor rank as measure of complexity

As discussed in the beginning of the post, a major challenge towards understanding implicit regularization in deep learning is that we lack measures for predictor complexity that capture natural data. 
Now, let us recap what we have seen:
$(1)$ tensor completion is equivalent to multi-dimensional prediction; 
$(2)$ tensor factorization corresponds to solving the prediction task with a certain non-linear neural network; 
and 
$(3)$ the implicit regularization of this non-linear network, i.e. of tensor factorization, minimizes tensor rank.
Motivated by these findings, we ask the following:

> **Question:** 
> Can tensor rank serve as a measure of predictor complexity?

We empirically explored this prospect by evaluating the extent to which tensor rank captures natural data, i.e. to which natural data can be fit with predictors of low tensor rank.
As testbeds we used [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) datasets, comparing the resulting errors against those obtained when fitting two randomized variants: one generated via shuffling labels (‘‘rand label''), and the other by replacing inputs with noise (‘‘rand image'').

The following plot, displaying results on Fashion-MNIST (those on MNIST are similar), shows that with predictors of low tensor rank the original data is fit far more accurately than the randomized datasets. 
Specifically, even with tensor rank as low as one the original data is fit relatively accurately, whereas the error in fitting random data is close to trivial (variance of the label). 
This suggests that tensor rank as a measure of predictor complexity has potential to capture aspects of natural data! 
Note also that an accurate fit with low tensor rank coincides with low test error, which is not surprising given that low tensor rank predictors can be described with a small number of parameters.

<div style="text-align:center;">
<!--<img style="width:600px;padding-bottom:15px;padding-top:10px;" src="http://www.offconvex.org/assets/exp_complexity_fmnist.png" />-->
<img style="width:600px;padding-bottom:15px;padding-top:10px;" src="https://noamrazin.github.io/files/temp/exp_complexity_fmnist.png" />
<br>
<i><b>Figure 5:</b> 
Evaluation of tensor rank as a measure of complexity &mdash; standard datasets <br>
can be fit accurately with predictors of low tensor rank (far beneath what is required by <br>
random datasets), suggesting it may capture aspects of natural data. Plot shows mean <br>
error of predictors with low tensor rank over Fashion-MNIST. Markers correspond <br>
to separate runs differing in the explicit constraint on the tensor rank.
</i>
</div>

## Concluding thoughts

Overall, [our paper](https://arxiv.org/pdf/2102.09972.pdf) shows that tensor rank captures both the implicit regularization of a certain non-linear neural network, and aspects of natural data. 
In light of this, we believe that tensor rank (or more advanced notions such as hierarchical tensor rank) may pave way to explaining both implicit regularization in more elaborate neural networks, and the properties of real-world data translating this implicit regularization to generalization.

[Noam Razin](https://noamrazin.github.io/), [Asaf Maman](https://asafmaman101.github.io/), [Nadav Cohen](http://www.cohennadav.com/)
