---
layout:     post
title:      Equilibria in nonconvex-nonconcave min-max optimization  
date:       2020-06-24 15:00:00
summary:    Present a new notion of local equilibrium in general min-max optimization. 
author:     Oren Mangoubi and Nisheeth Vishnoi
visible:    False
---

While there has been incredible progress in convex and nonconvex minimization, a multitude of problems in ML today are in need of efficient algorithms to solve min-max optimization problems. 
 Unlike minimization, where algorithms can always be shown to converge to some local minimum, there is no notion of a local equilibrium in min-max optimization that exists for general nonconvex-nonconcave functions.
    In two recent papers, we give  two notions of local equilibria that are guaranteed to exist and efficient algorithms to compute them.
In this post we present the key ideas behind a second-order notion of local min-max equilibrium from [this paper](https://arxiv.org/abs/2006.12363) and in the next we will talk about a different notion along with the algorithm and show its implications to GANs from [this paper](https://arxiv.org/abs/2006.12376). 



## Min-max optimization

Min-max optimization of an objective function $f:\mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$
$$  \min_x \max_y f(x,y),$$
is a powerful framework in optimization, economics, and ML as it allows one to model learning in the presence of multiple agents with competing objectives.
In ML applications, such as [GANs](https://arxiv.org/abs/1406.2661)and [adversarial robustness](https://adversarial-ml-tutorial.org), the min-max objective function may be nonconvex-nonconcave.
We know that min-max optimization is at least as hard as minimization, hence, we cannot hope to find a globally optimal solution to min-max problems for general functions.


## Approximate local minima for minimization

Let us first revisit the special case of minimization, where there is a natural notion of an approximate second-order local minimum.

>$x$ is a second-order $\varepsilon$-local minimum of $\mathcal{L}:\mathbb{R}^d\rightarrow \mathbb{R}$ if
$$    \|\nabla \mathcal{L}(x)\| \leq \varepsilon  \ \ \mathrm{and} \ \  \nabla^2 \mathcal{L}(x) \succeq -\sqrt{\varepsilon}. $$




<div style="text-align:center;">
<img src="/assets/GDA_spiral_fast.gif" alt="" />
<br>
<b>Figure 1.</b> GDA spirals off to infinity from almost every starting point on the objective function $f(x,y) = xy$. 
</div>
<br />



