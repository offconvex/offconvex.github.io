---
layout:     post
title:      More on saddle points
date:       2016-03-24 9:00:00
summary:    Why it is hard to get stuck on saddle points
author:     Benjamin Recht
visible:    false
---

Thanks to Rong for the [very nice blog post](http://www.offconvex.org/2016/03/22/saddlepoints/) describing critical points of nonconvex functions and how to avoid them. I’d like to follow up on his post to highlight a fact that is not widely appreciated in nonlinear optimization: it is super hard to converge to a saddle point. (Just look at those pictures in Rong’s post!  If you move ever so slightly you fall off the saddle).  Even dumb algorithms like gradient descent with constant step sizes can’t converge to saddle points unless you try really hard.

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

$$
	x_{i}^{(k)} = (1-t a_i)^{k} x_i^{(0)}\,.
$$


One can immediately see from this expression that if the step size $t$ is chosen such 
that $t |a_i| < 1 $ 
for all $i$, then when all of the $a_i$ 
are nonnegative, the algorithm converges to a point where the gradient is equal to zero from any starting point.  But if there is *a single negative $a_i$*, the function diverges to negative infinity exponentially quickly from any randomly chosen starting point.

The random initialization is key here.  If we initialized the problem such that $x^{(0)}_i=0$ whenever $a_i<0$, then the algorithm would actually converge.  However, under the smallest perturbation away from this initial condition, gradient descent diverges to negative infinity.

Most of the examples showing that algorithms converge to stationary points are fragile in a similar way.  You have to try very hard to make an algorithm converge to a saddle point.  As an example of this phenomena for a non-quadratic function, consider the following example from Nesterov’s revered [Introductory Lectures on Convex Optimization](http://www.springer.com/us/book/9781402075537). Let $f(x,y) = \frac12 x^2 +\frac14 y^4-\frac12 y^2$.  The critical points of this function are $z^{(1)}= (0,0)$, $z^{(2)} = (0,-1)$ and $z^{(3)} = (0,1)$.  The points $z^{(2)}$ and $z^{(3)}$ are local minima, and $z^{(1)}$ is a saddle point.  Now observe that gradient descent initialized from any point of the form $(x,0)$ converges to the saddle point $z^{(1)}$. *Any other initial point* either diverges or converges to a local minimum.  If one chooses an initial point at random, then gradient descent does not converge to a saddle point *with probability one.*

### The Stable Manifold Theorem and random initialization
 
In recent work with [Jason Lee, Max Simchowitz, and Mike Jordan](http://arxiv.org/abs/1602.04915), we made this result precise using the Stable Manifold Theorem from dynamical systems.  The Stable Manifold theorem is concerned with fixed point operations of the form $x^{(k+1)} = \Psi(x^{(k)})$.  It quantifies that the set of points that locally converge to a fixed point $x^{\star}$ of such an iteration have measure zero whenever the Jacobian of $\Psi$ at $x^{\star}$ has eigenvalues bigger than 1.

With a fairly straightforward argument, we were able to show that the gradient descent algorithm satisfied the assumptions of the Stable Manifold Theorem, and, moreover, that the set of points that converge to strict saddles *always* has measure zero.  This formalizes the above argument.  If you pick a point at random and run gradient descent, you will never converge to a saddle point.

In some sense, optimizers would not be particularly surprised by this theorem.  We are sure that some version of our result is already known for gradient descent, but we couldn't find it in the literature.  If you can find an earlier reference proving this theorem we would be delighted if you’d let us know.

### Adding noise

As Rong [discussed](http://www.offconvex.org/2016/03/22/saddlepoints/), in his paper with Huang, Jin, and Yuan, adding gaussian noise to the gradient helps to avoid saddle points.  In particular, they introduce the notion *strict saddle* functions to be those where all saddle points are either local minima or have Hessians with negative eigenvalues bounded away from 0.  As we saw above, if a saddle point has negative eigenvalues, the set of initial conditions that converge to that point has measure zero.  But when we add noise to the gradient, there are *no initial conditions* that converge to saddles.  The noise immediately pushes you off this low-dimensional manifold. 

Interestingly, a similar result also follows from the Stable Manifold Theorem. Indeed, Robin Pemantle [developed an even more general result](https://www.math.upenn.edu/~pemantle/papers/nonconvergence.pdf) for stochastic processes.  Pemantle uses the Stable Manifold Theorem to show that general vector flows perturbed by noise cannot converge to unstable fixed points. As a special case, he proves that stochastic gradient descent cannot converge to a saddle point provided the gradient noise is sufficiently diverse.  In particular, this implies that additive gaussian noise is sufficient to prevent convergence to saddles.

Pemantle does not even have to assume the strict saddle point condition to prove his theorem.  However, additional work would be required to extract the sort of quantitative convergence bounds that Rong and his coauthors derive from Pemantle’s argument.

## What makes nonconvex optimization difficult?

If saddle points are easy to avoid, then the question remains as to what exactly makes nonconvex optimization difficult?  In my next post, I’ll explore why this question is so challenging, describing some apparently innocuous problems in optimization that are deviously difficult.
