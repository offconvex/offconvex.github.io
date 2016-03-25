---
layout:     post
title:      More on saddle points
date:       2016-03-24 9:00:00
summary:    Why it is hard to get stuck on saddle points
author:     Benjamin Recht
visible:    false
---

Thanks to Rong for the [very nice blog post](http://www.offconvex.org/2016/03/22/saddlepoints/) describing critical points of nonconvex functions and how to avoid them. In this post, I’d like to highlight a couple of facts that are not widely appreciated about nonlinear optimization. First, it is super hard to converge to a saddle point. (Just look at those pictures in Rong’s post!  If you move ever so slightly you fall off the saddle).  Dumb algorithms like gradient descent with constant step sizes can’t converge to saddle points unless you try really hard.  Second, and more importantly, I would like to briefly describe why the hardness of non-convex optimization likely doesn’t arise from strict saddle points but from more subtle degenerate critical phenomena.

## It’s hard to converge to a saddle.

To illustrate why gradient descent would not converge to a non-minimizing saddle points, consider the case of a non-convex quadratic, $f(x)=\frac{1}{2} \sum_{i=1}^d a_i x_i^2$.  Assume that $a_i$ is nonnegative for the $k$ values and is strictly negative for the last $d-k$ values.  The unique stationary point of this problem is $x=0$.  The Hessian at $0$ is simply the diagonal matrix with $H_{ii} = a_i$ for $i=1,\ldots,d$.  

Now what happens when we run gradient descent on this function from some initial point $x^{(0)}?$  The gradient method has iterates of the form

$$
	x^{(k+1)} = x^{(k)} - t \nabla f(x^{(k)})\,.
$$

For our function, this takes the form

$$
	x^{(k+1)}_i = (1- t a_i) x_i^{(k)}
$$
If one unrolls this recursive formula down to zero, we see that the $i$th coordinate of the $k$th iterate is given by the formula

\[
	x_{i}^{(k)} = (1-t a_i)^{k} x_i^{(0)}\,.
\]


One can immediately see from this expression that if the step size $t$ is chosen such 
that $t |a_i| < 1 $ 
for all $i$, then when all of the $a_i$ 
are nonnegative, the algorithm converges to a point where the gradient is equal to zero from any starting point.  But if there is *a single negative $a_i$*, the function diverges to negative infinity exponentially quickly from any randomly chosen starting point.

The random initialization is key here.  If we initialized the problem such that $x^{(0)}_i=0$ whenever $a_i<0$, then the algorithm would actually converge.  However, under the smallest perturbation away from this initial condition, gradient descent diverges to negative infinity.

Most of the examples showing that algorithms converge to stationary points are fragile in a similar way.  You have to try very hard to make an algorithm converge to a saddle point!  As an example of this phenomena for a non-quadratic function, consider the following example from Nesterov’s revered [Introductory Lectures on Convex Optimization](http://www.springer.com/us/book/9781402075537). Let $f(x,y) = \frac12 x^2 +\frac14 y^4-\frac12 y^2$.  The critical points of this function are $z^{(1)}= (0,0)$, $z^{(2)} = (0,-1)$ and $z^{(3)} = (0,1)$.  The points $z^{(2)}$ and $z^{(3)}$ are local minima, and $z^{(1)}$ is a saddle point.  Now observe that gradient descent initialized from any point of the form $(x,0)$ converges to the saddle point $z^{(1)}$. *Any other initial point* either diverges or converges to a local minimum.  If one chooses an initial point at random, then gradient descent does not converge to a saddle point *with probability one.*

### The Stable Manifold Theorem and random initialization
 
In recent work with [Jason Lee, Max Simchowitz, and Mike Jordan](http://arxiv.org/abs/1602.04915), we made this result precise using the Stable Manifold Theorem from dynamical systems.  The Stable Manifold theorem is concerned with fixed point operations of the form $x^{(k+1)} = \Psi(x^{(k)})$.  It quantifies that the set of points that locally converge to a fixed point $x^{\star}$ of such an iteration have measure zero whenever the Jacobian of $\Psi$ at $x^{\star}$ has eigenvalues bigger than 1.

With a fairly straightforward argument, we were able to show that the gradient descent algorithm satisfied the assumptions of the Stable Manifold Theorem, and, moreover, that the set of points that converge to strict saddles *always* has measure zero.  This formalizes the above argument.  If you pick a point at random and run gradient descent, you will never converge to a saddle point.

In some sense, optimizers would not be particularly surprised by this theorem.  We are sure that some version of our result is already known for gradient descent, but we couldn't find it in the literature.  If you can find an earlier reference proving this theorem we would be delighted if you’d let us know!

### Adding noise

As Rong [discussed](http://www.offconvex.org/2016/03/22/saddlepoints/), in his paper with Huang and Jin, they show that adding gaussian noise to the gradient helps to avoid saddle points.  In particular, they introduce the notion *strict saddle* functions to be those where all saddle points are either local minima or have Hessians with negative eigenvalues bounded away from 0.  As we saw above, once you have negative eigenvalues, you’ll never converge to a fixed point.  But adding noise to the gradient makes it completely impossible to converge to a saddle.

A considerably more general result for stochastic processes is developed in [this beautiful paper](https://www.math.upenn.edu/~pemantle/papers/nonconvergence.pdf) by Robin Pemantle.  Pemantle uses the Stable Manifold Theorem to show that general vector flows perturbed by noise cannot converge to unstable fixed points. As a special case, he proves that stochastic gradient descent cannot converge to a saddle point provided the gradient noise is sufficiently diverse.  In particular, this implies that additive gaussian noise is sufficient to prevent convergence to saddles.

When one adds noise to the gradient, there are *no initial conditions* that converge to saddles.  By no initial conditions, I mean that there isn’t even a set of measure zero that converges to a saddle point any more, because the noise immediately pushes you off this low-dimensional manifold. 

Pemantle does not even have to assume the strict saddle point condition to prove his theorem.  However, if one assumes that all saddle points are strict, one can extract quantitative convergence bounds from his proof.

## What makes nonconvex optimization difficult?

If saddle points are easy to avoid, then the question remains as to what exactly makes nonconvex optimization difficult?  First, let me say that it’s a bit ridiculous to define a class of problems using the “non-” prefix.  Defining a research area as the complement of a small class is going to always be far too general.

A reasonable approximation of what most people envision when they use the word “non-convex” is optimizing functions that are twice differentiable over simple sets.  Let’s call this “smooth optimization with convex constraints.”  While the relu and max-pooling units inside modern neural networks violate my assumptions, this is a good starting place because unconstrained smooth optimization is deceptively difficult!

My favorite nonconvex function class is the homogeneous quartics.  These functions are infinitely differentiable and yet incredibly difficult to optimize even in practice.  Consider the simple set of instances on $\mathbb{R}^d$
$$
	f(x) = \sum_{i,j=1}^d Q_{ij} x_i^2 x_j^2
$$
Where $Q$ is some $d\times d$ matrix.  When $Q$ is positive semidefinite, then $0$ is a global minimizer of $f$.  Indeed, it’s easy to see in that case that $f$ is nonnegative everywhere, so any point where $f(x)=0$ is a global optimum.  When $Q$ has only nonnegative entries, the same argument applies.  Indeed, generalizing these two cases, it’s easy to see that if $Q$ is the sum of a positive definite matrix and a matrix with only nonnegative entries, then $0$ is a global minimizer.

Now, what about for general $Q$?  Amusingly, we always have that the gradient vanishes at $0$.  So is $0$ a local minimum, a global minimum, a local maximum, a global maximum, or a saddle point?  Since $f(x)$ is homogenous in the sense that $f(tx) = t^4 f(x)$ for all scalars $t$, the only cases that can occur is that $0$ a global minimum, a global maximum, or a saddle point.  But any  of these cases can and do occur!

So can we check if $0$ is a global minimizer?  Well, $0$ is not a global minimizer if and only if there exists some $x$ such that $f(x)<0$.  If we perform the variable substitution, $u_i = x_i^2$, we see that $0$ is not a global minimizer if and only if there exists some $u$ with only nonnegative entries such that $u^T Q u <0$.  That is, $x$ is a global minimizer of $f$ if and only if the matrix $Q$ is *copositive* (a matrix is copositive if $u^T Q u \geq 0$ for all $u$ with only nonnegative entries).

Now, here’s the tricky part.  Checking if $Q$ is a copositive matrix is NP-hard.  Indeed, it’s really easy to encode hard problems into checking copositivity.  A relatively simple example is cleanly described in [expository lecture notes](http://www.ti.inf.ethz.ch/ew/lehre/ApproxSDP09/notes/copositive.pdf) by Bernd Gaertner and Jiri Matousek.  Let $G$ be a graph with $d$ vertices. Let $A$ be the adjacency matrix of $G$. Let $I$ denote the $d\times d$ identity matrix and $E$ the $d\times d$ matrix of all ones.  Let $s$ be some positive number.  Set
$$
	Q = I + A - s\cdot E
$$
Then $G$ has an independent set of size larger than $1/s$ if and only if $Q$ is not copositive (this is Theorem 5.2.5 in Gaertner and Matousek’s notes).  In other words, finding a direction to decrease the function value of a quartic polynomial is at least as hard as deciding if a graph has an independent set of a specified size.

Now, I know my theory friends are going to come back to me and say that random instances of independent set can be solved by greedy heuristics.  I used independent set because it had the simplest formulation as a smooth optimization problem.  However, there are also reductions from [Maximum Clique](http://www.caam.rice.edu/~yad1/miscellaneous/References/Math/Topology/Cliques/Maximal%20Clique%20Problem.pdf) and [Subset Sum](http://www-personal.umich.edu/~murty/np.pdf) to quartic minimization which are almost as simple.  And, moreover, note that “adding noise to the gradient” can’t provide you a particularly good heuristic to find a descent direction.  This would amount to solving Subset Sum by random guessing.  Good luck with that!

So what makes this case particularly hard?  It’s that the Hessian of $f$ at zero is the all zeros matrix.  The Hessian gives us no information whatsoever about the curvature of $f$.  In retrospect, this shouldn’t be too surprising.  $0$ is a stationary point of the functions $x^3$ and $x^4$, and the second derivatives of both functions vanish at zero.  But, of course, in one case there is a descent direction and in the other there is not.

However, note that negative eigenvalues have nothing to do with the hardness here.

## Is non-convex optimization hard in practice?

One of the surprisingly open questions in non-convex optimization is getting a handle on whether non-smooth optimization is generically hard or generically easy.  As a counterpoint, consider the traveling salesman problem.  We know that TSP is hard to approximate, but [there is free software](http://www.math.uwaterloo.ca/tsp/concorde.html) that will quickly, *exactly* solve nearly any real instance that arises in actual path planning or circuit routing.

For nonsmooth optimization, there is not as clear of a picture.  For an argument that nonsmooth optimization quite hard in practice, go browse the [Journal of Global Optimization](http://www.springer.com/mathematics/journal/10898) for a catalog of daunting problems.  On the other hand, for an argument that nonconvexity is easy, take a look at proceedings of [ICLR](http://www.iclr.cc/doku.php).  Certainly the answer lies is in the middle.  There are problems that are easy to formulate as unconstrained nonlinear problems and then solve with local search, and there are problems which are hard no matter how you formulate them.    As this post illustrates, we have a long way to go before we reach a consensus on whence nonconvexity inherits its hardness.  But an exciting challenge to the machine learning community is to determine just what makes a typical learning problem easy and to build tools that solve easy instances as efficiently as possible.
