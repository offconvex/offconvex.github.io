---
layout:     post
title:      A Framework for analysing Non-Convex Optimization.
date:       2016-05-08 9:00:00
summary:    Sufficient Conditions for Fast Convergence to Global Minimum
author:     Sanjeev Arora, Tengyu Ma
visible:    True
---


Previous blog posts of [Rong](http://www.offconvex.org/2016/03/22/saddlepoints/) and [Ben](http://www.offconvex.org/2016/03/24/saddles-again/) show that (noisy) gradient descent can converge to *local* minimum of a non-convex function, and in (large) polynomial time ([Ge et al.’15](http://arxiv.org/abs/1503.02101)). This post 
describes a simple framework that can sometimes be used to design/analyse algorithms that can quickly reach an approximate *global* optimum of the nonconvex function. The framework ---which was used to analyse alternating minimization algorithms for sparse coding  in [our COLT'15 paper with Ge and Moitra](http://arxiv.org/abs/1503.00778)---generalizes many other sufficient conditions for convergence (usually gradient-based) that were formulated in recent papers.

##Measuring progress: a simple Lyapunov function 

Let $f$ to be the function being optimized and suppose the algorithm produces a sequence of candidate solutions $z_1,\dots,z_k,\dots,$ via some update rule
$$z_{k+1} = z_k - \eta g_k.$$


This can be seen as a dynamical system (see [Nisheeth’s](http://www.offconvex.org/2016/04/04/markov-chains-dynamical-systems/) and [Ben’s](http://www.offconvex.org/2016/03/24/saddles-again/) posts related to dynamical systems).
Our goal is to show that this sequence converges to (or gets close to) a target point $z^* $, which is  a global optimum of $f$. Of course, the algorithm doesn't know $z^*$.

To design a framework for proving convergence it helps to indulge in daydreaming/wishful thinking: what property would we *like* the updates to have, to simplify our job? 

A natural idea is to define a Lyapunov function $V(z)$ and show that: (i) $V(z_k)$ decreases to $0$ (at a certain speed) as $k\rightarrow \infty$; (ii) when $V(z)$ is close to $0$, then $z$ is close to $z^* $. (Aside: One can imagine more complicated ways of proving convergence, e.g., show $V(z_k)$ ultimately goes to $0$ even though it doesn't decrease in every step. Nesterov's acceleration method uses such a progress measure.)

Consider possibly the most trivial Lyapunov function, the (squared) distance to the target point,  $V(z) = \|z-z^*\|^2$. This is also used in the standard convergence proof for convex functions, since moving in the opposite direction to the gradient can be shown to reduce this measure $V()$. 
 
Even when the function is nonconvex, there always *exist* update directions that reduce this $V()$ (though finding them may not be easy).  Simple algebraic manipulation shows that when the learning rate $\eta$ is small enough, then for $V(z_{k+1}) \le V(z_k)$, it is necessary and sufficient to have $\langle g_k, z_k-z^* \rangle \ge 0$. 

  <img src="http://www.cs.princeton.edu/~tengyu/angle_for_blog_post.png" alt="correlation condition" style ="float:left" height="240.8" width="260"/> As illustrated in the figure on the left, $z^* - z_k$ is the ideal direction that we desire to move to, and $-g_k$ is the direction that we actually move to. To establish convergence, it suffices to verify that the direction of movement is positively correlated with the desired direction. 

To get quantitative bounds on running time, we need to ensure that $V(z_k)$ not only decreases, but does so rapidly. The next condition formalizes this: intuitively speaking it says that $-g_k$ and $z^*-z_k$ make an angle strictly less than 90 degrees. 

> **Correlation Condition**: The direction $g_k$ is $(\alpha,\beta,\epsilon_k)$-correlated with $ z^* $  if 
$$\langle g_k,z_k-z^* \rangle \ge \alpha \|z_k-z^*\|^2 + \beta \|g_k\|^2 -\epsilon_k$$

This may look familiar to experts in convex optimization: as a special case if we make the update direction $g_k$ stand for the (negative) gradient, then the condition yields familiar notions such as strong convexity and smoothness. But the condition allows $g_k$ to not be the gradient, and in addition, allows  the error term $\epsilon_k$, which is necessary in some applications to accommodate non-convexity and/or statistical error. 

If the algorithm can at each step find such update directions, then the familiar convergence proof of convex optimization can be modified to show rapid convergence here as well, except the convergence is *approximate*, to some point in the neighborhood of $z^*$.


> **Theorem:** Suppose $g_k$ satisfies the Correlation Condition above for every $k$, then with learning rate $\eta \le 2\beta$, we have 
$$\| z_k-z^* \|^2 \le (1-\alpha\eta)^k\| z_0-z^* \|^2 + \max_k \epsilon_k/\alpha$$


###Comparison with related conditions 

As mentioned, the "wishful thinking" approach has been used to identify other 
conditions under which specific nonconvex optimizations can be carried out to near-optimality: ([JNS’13](https://arxiv.org/abs/1212.0467), [Hardt’14](http://arxiv.org/abs/1312.0925), [BWY’14](http://arxiv.org/abs/1408.2156), [CLS’15](http://arxiv.org/abs/1407.1065), [AGMM’15](http://arxiv.org/abs/1503.00778), [SL’15](http://arxiv.org/abs/1411.8003), [CC’15](http://arxiv.org/abs/1505.05114), [ZWL’15](https://papers.nips.cc/paper/5733-a-nonconvex-optimization-framework-for-low-rank-matrix-estimation)). All of these can be seen as some weakening of convexity (with the exception of the analysis for matrix completion in [Hardt’14](http://arxiv.org/abs/1312.0925) which views the updates as noisy power method). 

Our condition appears to contain most if not all of these as special cases. 

Often the update direction $g_k$ in these papers is related to the gradient. For example using the gradient instead of 
$g_k$ in our correlation condition turns it into the "regularity condition" proposed by [CLS’15](http://arxiv.org/abs/1407.1065) for analyzing Wirtinger flow algorithm for phase retrieval. 
The gradient stability condition in [BWY’14](http://arxiv.org/abs/1408.2156) is also a special case, where $g_k$ is required to be close enough to $\nabla h(z_k)$ for some convex $h$ such that $z^* $ is the optimum of $h$. Then since $\nabla h(z_k)$ has angle < 90 degrees with $z_k-z^*$ (which follows from convexity of $h$), it implies that $g_k$ also does.  

The advantage of our framework is that it encourages one to think of algorithms where $g_k$ is not the gradient.  Thus applying the framework doesn't require understanding the behavior of the gradient on the entire landscape of the objective function; instead, one needs to understand the update direction (which is  under the algorithm designer's control) at the 
sequence of points actually encountered while running the algorithm.

This slight change of perspective may be powerful.



##Application to Sparse Coding

A particularly useful situation for applying the framework above is where the objective function has two sets of arguments and it is feasible to optimize one set after fixing the other --leading to the familiar alternating minimization heuristic. Such algorithms are a good example of how one may try to do local-improvement without explicitly following the (full) gradient. 
As mentioned, our framework was used to analyse
such alternating minimization for [sparse coding](https://en.wikipedia.org/wiki/Neural_coding#Sparse_coding). 

In sparse coding, we are given a set of examples $Y = [y_1,\dots, y_N]\in \mathbb{R}^{d\times N}$, and are asked to find an over-complete basis $A = [a_1,\dots,a_m]$  (where "overcomplete" refers to the setting
$m > d$) so that each example $y_j$ can be expressed as a sparse linear combination of $a_i$’s. Therefore, the natural optimization problem with squared loss is that 

$$f(A,X) = \min_{A, \textrm{sparse } X} \|Y-AX\|^2$$

Here both the objective and the constraint set are not convex. 
One could consider using $\ell_1$ regularization as a surrogate for sparsity, but the trouble will be that the regularization is neither smooth or strongly convex, and the standard techniques for dealing with $\ell_1$ penalty term in convex optimization cannot be easily applied due to  non-convexity. 

The standard alternating minimization  algorithm (a close variant of the one proposed by [Olshausen and Field 1997](http://redwood.psych.cornell.edu/papers/olshausen_field_1997.pdf) as a neurally plausible explanation for V1, the human primary visual cortex) is as follows: 

$$
X_{k+1} \longleftarrow \textrm{threshold}(A_k^{\top}Y) 
$$

$$
A_{k+1} \longleftarrow A_{k} - \eta \underbrace{\frac{\partial }{\partial A} f(A_k,X_{k+1})}_{G_k}
$$

Here update for $X$ is the projection pursuit algorithm in sparse recovery (see [Elad’10](http://www.springer.com/us/book/9781441970107) for background), which is supposed to give an approximation of the best fit for $X$ given the current $A$. 

Sometimes alternating minimization algorithms need careful initialization, but in practice here it suffices to initialize $A_0$ using a random sample of datapoints $y_i$'s. 

However, it remains an open problem to analyse convergence using such random initialization; our analysis uses a special starting point $A_0$ found using spectral methods.

### Applying our framework

At first glance, the mysterious aspect of our framework was how the algorithm can find an update direction correlating with $z_k -z^* $,  without knowing $z^* $? In context of sparse coding, this comes about as follows:  if we assume a probabilistic generative model for the observed data (namely, it was generated using some ground-truth sparse coding) then the alternating minimization automatically comes up with such  update directions!


Specifically, we will assume that the data points $y_i$'s are generated using some ground truth dictionary $A^* $ 
 using some ground truth $X^* $ whose columns are iid draws from some suitable distribution.
 (One needs to assume some conditions on $A^* , X^* $, which  are not important in the sketch below.)  Note that the 
 entries within each column of $X^* $  are *not* mutually independent, otherwise the problem would be [Independent Component Analysis](https://en.wikipedia.org/wiki/Independent_component_analysis). 
 
 
In line with our framework, we consider the Lyapunov function $V(A) = \|A-A^* \|_F^2$. Here the Frobenius norm $\|\cdot\|_F$ is also the Euclidean norm of the vectorized version of the matrix. Then our framework implies that to show quick convergence it suffices to verify the following (for some $\alpha,\beta > 0$)	 for the update direction $G_k$:

$$\langle G_k, A_k-A^* \rangle \ge \alpha \|A_k - A^* \|_F^2 + \beta \|G_k\|_F^2 -\epsilon_k $$

In [AGMM’15](http://arxiv.org/abs/1503.00778) we showed that under certain assumption on the true dictionary $A^* $ and the true coefficient $X^* $, the above inequality is indeed true with small $\epsilon_k$ and some constant $\alpha,\beta > 0$. The proof is a bit technical but reasonable — the partial gradient $\frac{\partial f}{\partial A}$ has a simple form and therefore $G_k$ has a closed form in $A_k$ and $Y$. Therefore, it boils down to plugging in the form of $G_k$ into the equation above and simplifying it appropriately.  (One also needs the fact that
the starting $A_0$ obtained using spectral methods is somewhat close to $A^* $.)  

We hope others will use our framework to analyse other nonconvex problems!

*(Aside: We hope that readers will leave comments if they know of other frameworks for proving convergence that are not subcases of the above framework.)*

