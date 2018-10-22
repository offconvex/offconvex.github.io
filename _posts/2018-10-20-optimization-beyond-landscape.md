---
layout: post
title: Do landscape analyses suffice for understanding optimization in deep learning?
date: 2018-3-2 13:00:00
author: Nadav Cohen
visible: True
---

Neural network optimization is fundamentally non-convex, and yet simple gradient-based algorithms seem to consistently solve such problems.
This phenomenon is one of the central pillars of deep learning, and forms a mystery many of us theorists are trying to unravel.


## Landscape Approach

By far the most popular approach for theoretical analysis of optimization in deep learning is based on *landscapes*.
In a nutshell, the idea is to characterize the geometry of the training objective independently of the optimization algorithm, and then show, using very general properties of the algorithm (for example that it locally descends or that it is stochastic), that convergence (hopefully to global minimum) takes place.
One geometric condition established in various settings (see for example [TODO: ADD REFS]) is referred to as *"no poor local minima"*, and means that all local minima are close in their objective values to a global minimum.

[TODO: ADD ILLUSTRATION]

The "no poor local minima" property does not guarantee convergence to global minimum of gradient-based algorithms, as these may get stuck around *saddle points*, i.e. points where the gradient of the objective function vanishes but are not a local minimum.
To disqualify this scenario, another geometric property was defined: "strict saddle".
Formally, this property means that every saddle point has at least one negative eigenvalue to its Hessian.
Informally, it means that at every saddle point, there exists at least one direction of movement that would lead local descent algorithms to quickly slide down from the saddle.

[TODO: ADD ILLUSTRATION]

Several works (e.g. [TODO: ADD REFS]) have established, in different technical settings, that for an objective landscape admitting both "no poor local minima" and "strict saddle" properties, a gradient-based algorithm will converge to global minimum.
These results have been successfully applied to non-convex problems such as tensor decomposition ([TODO: ADD REF]), matrix completion ([TODO: ADD REF]), matrix factorization ([TODO: ADD REF]), matrix sensing ([TODO: ADD REF]), phase retrieval ([TODO: ADD REF]), dictionary learning  ([TODO: ADD REF]) and two layer neural networks with quadratic activation ([TODO: ADD REF]).


## Limitations

The aforementioned problems in which the landscape approach has successfully proven convergence to global minimum for a gradient-based algorithm, can all be viewed as incarnations of *shallow* (two layer) models.
The absence of *deep* (three or more layer) networks in the portfolio of success stories is not coincidental --- unfortunately, even the simplest of such models violate the "strict saddle" property (see [Kawaguchi 2016](https://papers.nips.cc/paper/6112-deep-learning-without-poor-local-minima.pdf) for formal examples).
Consider for example the loss for a fully-connected neural network, at the point where all weights are set to zero (ignore biases for simplicity).
There, if the depth (number of layers) is two or more, the gradient will vanish, and if it is three or more, the Hessian will vanish as well.
Accordingly, unless this point is a local minimum, which will not be the case in any reasonable setting, we obtain a saddle without any negative eigenvalue to its Hessian.

One may argue that violation of the "strict saddle" property by deep models does not necessarily point to a limitation of the general landscape approach, but rather implies that we should come up with alternative geometric conditions that ensure tractable optimization while being met by deep networks.
In a [paper](http://proceedings.mlr.press/v80/arora18a.html) published last ICML (presented in a [previous blog post](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/), Sanjeev Arora, Elad Hazan and I argued that, sometimes, adding (redundant) linear layers to a classic linear model can accelerate gradient-based optimization, without any gain in expressiveness, and despite introducing non-convexity to a formerly convex problem.
Any landscape analysis that relies on properties of critical points (points where the gradient of the objective function vanishes) will have difficulty explaining this phenomenon, as through such lens, nothing is easier to optimize than a convex objective with a single critical point which is the global minimum.

Finally, perhaps the most potent evidence pointing to limitations of the landscape approach in the context of deep learning is the fact that, in many settings, proper initialization seems to play a crucial role in the success of neural network optimization.
This was shown by systematic empirical studies (e.g. [Sutskever et al. 2013](http://proceedings.mlr.press/v28/sutskever13.html)), and lately also by theory (as discussed below).
In its current spirit, the landscape approach typically decouples analysis of objective geometry from particularities of the optimization procedure, including the way it is initialized.
Capturing phenomena by which some initialization schemes result in successful optimization while others fail is somewhat contradictory to such decoupling.


## A Way Out?

The determining role of initialization in gradient-based optimization of deep networks suggests that perhaps a more relevant question than "is the landscape graceful?" is "what is the behavior of specific optimizer *trajectories* emanating from specific initializations?".

[TODO: ADD ILLUSTRATION]

While the trajectory-based approach is seemingly much more burdensome than landscape analyses, it is already leading to notable progress.
Several recent papers (e.g. [TODO: ADD REFS]) have adopted this strategy, successfully analyzing different types of shallow (two layer) models.
Moreover, trajectory-based analyses are beginning to set foot beyond the realm of the landscape approach --- for the case of linear neural networks, they have successfully established convergence of gradient descent to global minimum under *arbitrary depth*.


## Trajectory-Based Analyses for Deep Linear Neural Networks

Linear neural networks are fully-connected neural networks with linear (no) activation.
Specifically, a depth $N$ linear network with input dimension $d_0$, output dimension $d_N$, and hidden dimensions $d_1,d_2,\ldots,d_{N-1}$, is a linear mapping from $\mathbb{R}^{d_0}$ to $\mathbb{R}^{d_N}$ parameterized by $x \mapsto W_N W_{N-1} \cdots W_1 x$, where $W_j \in \mathbb{R}^{d_j \times d_{j-1}}$ is regarded as the weight matrix of layer $j$.
Though trivial from a representational perspective, linear neural networks are, somewhat surprisingly, complex in terms of optimization --- they lead to non-convex training problems with multiple minima and saddle points.
Being viewed as a theoretical surrogate for optimization in deep learning, the application of gradient-based algorithms to linear neural networks is receiving significant attention these days.

To my knowledge, [Saxe et al. 2014](https://arxiv.org/pdf/1312.6120.pdf) were the first to carry out a trajectory-based analysis for deep (three or more layer) linear networks, treating gradient flow (gradient descent with infinitesimally small learning rate) minimizing $\ell_2$ loss over whitened data.
Though a very significant contribution, this analysis did not formally establish convergence to global minimum, nor treat the aspect of computational complexity (number of iterations required to converge).
The recent work of [Bartlett et al. 2018](http://proceedings.mlr.press/v80/bartlett18a.html) makes progress towards addressing these gaps, by applying a trajectory-based analysis to gradient descent for the special case of linear residual networks, i.e. linear networks with uniform width across all layers ($d_0=d_1=\cdots=d_N$) and identity initialization ($W_j=I ~ \forall j$).
Considering different data-label distributions (which boil down to what they refer to as "targets"), Bartlett et al. demonstrate cases where gradient descent provably converges to global minimum at a linear rate --- loss is less than $\epsilon>0$ from optimum after $\mathcal{O}(\log\frac{1}{\epsilon})$ iterations --- as well as situations where it fails to converge.

In a [new paper](https://arxiv.org/pdf/1810.02281.pdf) with Sanjeev Arora, Noah Golowich and Wei Hu, we take an additional step forward in virtue of the trajectory-based approach.
Specifically, we analyze trajectories of gradient descent for any linear neural network that does not include "bottleneck layers", i.e. whose hidden dimensions are no smaller than the minimum between the input and output dimensions ($d_j \geq \min\{d_0,d_N\} ~ \forall j$), and prove convergence to global minimum, at a linear rate, provided that initialization meets the following two conditions:
(i) *approximate balancedness* --- $W_{j+1}^\top W_{j+1} \approx W_j W_j^\top ~ \forall j$;
and (ii) *deficiency margin* --- initial loss is smaller than the loss of any rank deficient solution.
We show that both conditions are necessary, in the sense that violating any one of them may lead to a trajectory that fails to converge.
Approximate balancedness at initialization is trivially met in the special case of linear residual networks, and also holds for the customary setting of initialization via small random perturbations centered at zero.
The latter also leads to deficiency margin with positive probability.
For the case $d_N=1$, i.e. scalar regression, we provide a random initialization scheme under which both conditions are met, and thus convergence to global minimum at linear rate takes place, with constant probability. 


## Conclusion

Tackling the question of optimization in deep learning through the landscape approach, i.e. by analyzing the geometry of the objective independently of the algorithm used for training, is conceptually appealing.
However, this strategy suffers from inherent limitations, predominantly as it requires the entire objective to be graceful, which seems to be too strict of a demand.
The alternative approach of taking into account the optimizer and its initialization, and focusing on the landscape only along the resulting trajectories, is gaining more and more traction.
While landscape analyses have thus far been limited to shallow (two layer) models only, the trajectory-based approach has recently treated arbitrarily deep models, proving convergence of gradient descent to global minimum at linear rate.
Much work however remains to be done, as this success covered only linear neural networks.
I expect the trajectory-based approach to be key in developing our formal understanding of gradient-based optimization for deep *non-linear* networks as well.


[Nadav Cohen](http://www.cohennadav.com/)
