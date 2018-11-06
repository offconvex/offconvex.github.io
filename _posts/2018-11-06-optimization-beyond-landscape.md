---
layout: post
title: Do landscape analyses suffice for understanding optimization in deep learning?
date: 2018-11-6 18:00:00
author: Nadav Cohen
visible: True
---

Neural network optimization is fundamentally non-convex, and yet simple gradient-based algorithms seem to consistently solve such problems.
This phenomenon is one of the central pillars of deep learning, and forms a mystery many of us theorists are trying to unravel. 
In this post I'll survey some recent attempts, and finish with a discussion on my [new paper with Sanjeev Arora, Noah Golowich and Wei Hu](https://arxiv.org/pdf/1810.02281.pdf), which for the case of gradient descent over deep linear neural networks, provides a guarantee for convergence to global minimum at a linear rate.


## Landscape Approach and Its Limitations

Many papers on optimization in deep learning implicitly assume that a rigorous understanding will follow from establishing geometric properties of the loss *landscape*, and in particular, of *critical points* (points where the gradient vanishes).
For example, through an analogy with the spherical spin-glass model from condensed matter physics, [Choromanska et al. 2015](http://proceedings.mlr.press/v38/choromanska15.pdf) argued for what has become a colloquial conjecture in deep learning:

> **Landscape Conjecture:**
> In neural network optimization problems, suboptimal critical points are very likely to have negative eigenvalues to their Hessian. 
> In other words, there are almost *no poor local minima*, and nearly all *saddle points are strict*. 

Strong forms of this conjecture were proven for loss landscapes of various simple problems involving **shallow** (two layer) models, e.g. [matrix sensing](https://papers.nips.cc/paper/6271-global-optimality-of-local-search-for-low-rank-matrix-recovery.pdf), [matrix completion](https://papers.nips.cc/paper/6048-matrix-completion-has-no-spurious-local-minimum.pdf), [orthogonal tensor decomposition](http://proceedings.mlr.press/v40/Ge15.pdf), [phase retrieval](https://arxiv.org/pdf/1602.06664.pdf), and [neural networks with quadratic activation](http://proceedings.mlr.press/v80/du18a/du18a.pdf).
There was also work on establishing convergence of gradient descent to global minimum when the Landscape Conjecture holds, as described in the excellent posts on this blog by [Rong Ge](http://www.offconvex.org/2016/03/22/saddlepoints/), [Ben Recht](http://www.offconvex.org/2016/03/24/saddles-again/) and [Chi Jin and Michael Jordan](http://www.offconvex.org/2017/07/19/saddle-efficiency/). 
They describe how gradient descent can arrive at a second order local minimum (critical point whose Hessian is positive semidefinite) by escaping all strict saddle points, and how this process is efficient given that perturbations are added to the algorithm. 
Note that under the Landscape Conjecture, i.e. when there are no poor local minima and non-strict saddles, second order local minima are also global minima.

<p style="text-align:center;">
<img src="/assets/optimization-beyond-landscape-points.png" width="100%" alt="Local minima and saddle points" />
</p>

However, it has become clear that the landscape approach (and the Landscape Conjecture) cannot be applied as is to **deep** (three or more layer) networks, for several reasons.
First, deep networks typically induce non-strict saddles (e.g. at the point where all weights are zero, see [Kawaguchi 2016](https://papers.nips.cc/paper/6112-deep-learning-without-poor-local-minima.pdf)).
Second, a landscape perspective largely ignores algorithmic aspects that empirically are known to greatly affect convergence with deep networks --- for example the [type of initialization](http://proceedings.mlr.press/v28/sutskever13.html), or [batch normalization](http://proceedings.mlr.press/v37/ioffe15.pdf).
Finally, as I argued in my [previous blog post](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/), based upon [work with Sanjeev Arora and Elad Hazan](http://proceedings.mlr.press/v80/arora18a/arora18a.pdf), adding (redundant) linear layers to a classic linear model can sometimes accelerate gradient-based optimization, without any gain in expressiveness, and despite introducing non-convexity to a formerly convex problem. 
Any landscape analysis that relies on properties of critical points alone will have difficulty explaining this phenomenon, as through such lens, nothing is easier to optimize than a convex objective with a single critical point which is the global minimum.


## A Way Out?

The limitations of the landscape approach for analyzing optimization in deep learning suggest that it may be abstracting away too many important details.
Perhaps a more relevant question than "is the landscape graceful?" is "what is the behavior of specific optimizer **trajectories** emanating from specific initializations?".

<p style="text-align:center;">
<img src="/assets/optimization-beyond-landscape-trajectories.png" width="66%" alt="Different trajectories lead to qualitatively different results" />
</p>

While the trajectory-based approach is seemingly much more burdensome than landscape analyses, it is already leading to notable progress.
Several recent papers (e.g. [Brutzkus and Globerson 2017](http://proceedings.mlr.press/v70/brutzkus17a/brutzkus17a.pdf); [Li and Yuan 2017](https://papers.nips.cc/paper/6662-convergence-analysis-of-two-layer-neural-networks-with-relu-activation.pdf); [Zhong et al. 2017](http://proceedings.mlr.press/v70/zhong17a/zhong17a.pdf); [Tian 2017](http://proceedings.mlr.press/v70/tian17a/tian17a.pdf); [Brutzkus et al. 2018](https://openreview.net/pdf?id=rJ33wwxRb); [Li et al. 2018](http://proceedings.mlr.press/v75/li18a/li18a.pdf); [Du et al. 2018](https://arxiv.org/pdf/1806.00900.pdf); [Liao et al. 2018](http://romaincouillet.hebfree.org/docs/conf/nips_GDD.pdf)) have adopted this strategy, successfully analyzing different types of shallow models.
Moreover, trajectory-based analyses are beginning to set foot beyond the realm of the landscape approach --- for the case of linear neural networks, they have successfully established convergence of gradient descent to global minimum under **arbitrary depth**.


## Trajectory-Based Analyses for Deep Linear Neural Networks

Linear neural networks are fully-connected neural networks with linear (no) activation.
Specifically, a depth $N$ linear network with input dimension $d_0$, output dimension $d_N$, and hidden dimensions $d_1,d_2,\ldots,d_{N-1}$, is a linear mapping from $\mathbb{R}^{d_0}$ to $\mathbb{R}^{d_N}$ parameterized by $x \mapsto W_N W_{N-1} \cdots W_1 x$, where $W_j \in \mathbb{R}^{d_j \times d_{j-1}}$ is regarded as the weight matrix of layer $j$.
Though trivial from a representational perspective, linear neural networks are, somewhat surprisingly, complex in terms of optimization --- they lead to non-convex training problems with multiple minima and saddle points.
Being viewed as a theoretical surrogate for optimization in deep learning, the application of gradient-based algorithms to linear neural networks is receiving significant attention these days.

To my knowledge, [Saxe et al. 2014](https://arxiv.org/pdf/1312.6120.pdf) were the first to carry out a trajectory-based analysis for deep (three or more layer) linear networks, treating gradient flow (gradient descent with infinitesimally small learning rate) minimizing $\ell_2$ loss over whitened data.
Though a very significant contribution, this analysis did not formally establish convergence to global minimum, nor treat the aspect of computational complexity (number of iterations required to converge).
The recent work of [Bartlett et al. 2018](http://proceedings.mlr.press/v80/bartlett18a.html) makes progress towards addressing these gaps, by applying a trajectory-based analysis to gradient descent for the special case of linear residual networks, i.e. linear networks with uniform width across all layers ($d_0=d_1=\cdots=d_N$) and identity initialization ($W_j=I$, $\forall j$).
Considering different data-label distributions (which boil down to what they refer to as "targets"), Bartlett et al. demonstrate cases where gradient descent provably converges to global minimum at a linear rate --- loss is less than $\epsilon>0$ from optimum after $\mathcal{O}(\log\frac{1}{\epsilon})$ iterations --- as well as situations where it fails to converge.

In a [new paper with Sanjeev Arora, Noah Golowich and Wei Hu](https://arxiv.org/pdf/1810.02281.pdf), we take an additional step forward in virtue of the trajectory-based approach.
Specifically, we analyze trajectories of gradient descent for any linear neural network that does not include "bottleneck layers", i.e. whose hidden dimensions are no smaller than the minimum between the input and output dimensions ($d_j \geq \min\{d_0,d_N\}$, $\forall j$), and prove convergence to global minimum, at a linear rate, provided that initialization meets the following two conditions:

> *(i)* **Approximate Balancedness:** $W_{j+1}^\top W_{j+1} \approx W_j W_j^\top$, $\forall j$

and

> *(ii)* **Deficiency Margin:** initial loss is smaller than the loss of any rank deficient solution.

We show that both conditions are necessary, in the sense that violating any one of them may lead to a trajectory that fails to converge.
Approximate balancedness at initialization is trivially met in the special case of linear residual networks, and also holds for the customary setting of initialization via small random perturbations centered at zero.
The latter also leads to deficiency margin with positive probability.
For the case $d_N=1$, i.e. scalar regression, we provide a random initialization scheme under which both conditions are met, and thus convergence to global minimum at linear rate takes place, with constant probability. 


## Conclusion

Tackling the question of optimization in deep learning through the landscape approach, i.e. by analyzing the geometry of the objective independently of the algorithm used for training, is conceptually appealing.
However this strategy suffers from inherent limitations, predominantly as it requires the entire objective to be graceful, which seems to be too strict of a demand.
The alternative approach of taking into account the optimizer and its initialization, and focusing on the landscape only along the resulting trajectories, is gaining more and more traction.
While landscape analyses have thus far been limited to shallow (two layer) models only, the trajectory-based approach has recently treated arbitrarily deep models, proving convergence of gradient descent to global minimum at a linear rate.
Much work however remains to be done, as this success covered only linear neural networks.
I expect the trajectory-based approach to be key in developing our formal understanding of gradient-based optimization for deep non-linear networks as well.


[Nadav Cohen](http://www.cohennadav.com/)
