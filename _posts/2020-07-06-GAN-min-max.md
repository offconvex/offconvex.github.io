---
layout:     post
title:      Training GANs From Theory to Practice
date:       2020-07-06 10:00:00
summary:    Present a new algorithm convergent min-max optimization algorithm for training GANs 
author:     Oren Mangoubi; Sushant Sachdeva; Nisheeth Vishnoi
visible:    False
---

GANs, that were originally discovered in the context of unsupervised learning, have had far reaching implications to science, engineering, and society. However, training GANs remains challenging (in part) due to the lack of convergent algorithms for nonconvex-nonconcave min-max optimization. In this post, we present a new first-order algorithm for min-max optimization which is particularly suited to GANs. This algorithm is guaranteed to converge to an equilibrium, is competitive in terms of time and memory with gradient descent-ascent and, most importantly, GANs trained on it seem to be stable.



## GANs and min-max optimization

Starting with the work of [Goodfellow et al.](http://papers.nips.cc/paper/5423-generative-adversarial-nets), Generative Adversarial Nets (GANs) have become a critical component in various ML systems; for prior posts on GANs, see [here](https://www.offconvex.org/2018/03/12/bigan/) for a post on  GAN architechture, and [here](https://www.offconvex.org/2017/03/15/GANs/) and [here](https://www.offconvex.org/2017/07/06/GANs3/) for posts which discuss  some of the many difficulties arising when training GANs. Mathematically, a GAN consists of a generator neural network $\mathcal{G}$ and a discriminator neural network $\mathcal{D}$ that are competing against each other in a way that, together,  learn the unknown distribution from which a given dataset arises.  The generator takes a random "noise" vector as input and maps this vector to a point, for instance an image. The discriminator takes points -- "fake" ones produced by the generator and "real" ones from the given dataset -- as inputs.  The discriminator then tries to classify these points as "real" or "fake". As a designer, we would like the generated points to be indistinguishable from those of the dataset. Thus, our goal is to choose weights $x$ for the generator network that allow it to generate points which are difficult for any discriminator to tell apart from real images. This leads to a min-max optimization problem where we look for weights $x$ which *minimize* the rate (measured by a loss function $f$) at which the discriminator correctly classifies the real and fake points. Simultaneously, we seek weights $y$ for the discriminator network which *maximize* this rate.

> **Min-max formulation of GANs** <br> <br> 
>
> $$\min_x \max_y f(x,y),$$
>
> $$f(x,y) :=  \mathbb{E}[ f_{\zeta, \xi}(x,y)],$$
>
> where $\zeta$ is a random point from the dataset, and $\xi \sim N(0,I_d)$ is a noise vector which the generator maps to a fake point.  $f_{\zeta, \xi}$ measures how accurately the discriminator $\mathcal{D}(y;\cdot)$ classifies $\zeta$ from $\mathcal{G}(x;\xi)$ produced by the generator using the input noise $\xi$.



In this formulation, there are several choices that we have to make as a GAN designer, and an important one is that of a loss function. One concrete choice is from the paper of Goodfellow et al.: the cross-entropy loss function:

$$f_{\zeta, \xi}(x,y) := \log(\mathcal{D}(y;\zeta)) + \log(1-\mathcal{D}(y;\mathcal{G}(x;\xi)))$$

See [here](https://machinelearningmastery.com/generative-adversarial-network-loss-functions/) for a summary and comparison of different loss functions.

Once we fix the loss function (and the architecture of the generator and discriminator), we can compute unbiased estimates of the value of $f$ and its gradients $\nabla_x f$ and $\nabla_y f$ using batches consisting of random Gaussian noise vectors $\xi_1,\ldots \xi_n \sim N(0,I_d)$ and random samples from the dataset $\zeta_1, \ldots, \zeta_n$.  For example, the stochastic batch gradient

$$  \frac{1}{n} \sum_{i=1}^n \nabla_x f_{\zeta_i, \xi_i}(x,y)$$

gives us an unbiased estimate for $\nabla_x f(x,y)$.

>But how do we solve the min-max optimization problem above using such a first-order access to $f$?



## Gradient descent-ascent and variants

Perhaps the simplest algorithm we can try for min-max optimization is gradient descent-ascent (GDA). As the generator wants to minimize with respect to $x$ and the discriminator wants to maximize with respect to $y$, the idea is to do descent steps for $x$ and ascent steps for $y$. How to do this is not clear as this is a sequential optimization problem, and one strategy is to let the generator and discriminator alternate:

$$x_{i+1} = x_i -\nabla_x f(x_i,y_i),$$

$$y_{i+1} = y_i +\nabla_y f(x_i,y_i).$$

Other variants include, for instance, [optimistic mirror descent](https://arxiv.org/abs/1311.1869) (OMD) (see also [here](https://arxiv.org/abs/1807.02629) and  [here](https://arxiv.org/abs/1711.00141) for applications of OMD to GANs, and [here](https://arxiv.org/abs/1901.08511) for an analysis of OMD and related methods)

$$x_{i+1} = x_i -2\nabla_x f(x_i,y_i) + \nabla_x f(x_{i-1},y_{i-1})$$ 

$$y_{i+1} = y_i +2\nabla_y f(x_i,y_i)- \nabla_y f(x_{i-1},y_{i-1}).$$

The advantage of such algorithms is that they are quite practical. The problem. as discussed next, is that they are not always guaranteed to converge. Most of these guarantees only hold for special classes of loss functions $f$ such as concavity (see [here](https://papers.nips.cc/paper/9430-efficient-algorithms-for-smooth-minimax-optimization.pdf) and [here](https://arxiv.org/abs/1906.00331)) or [monotonicity](https://papers.nips.cc/paper/9631-solving-a-class-of-non-convex-min-max-games-using-iterative-first-order-methods.pdf), or under the assumptions that these algorithms are provided with special starting points (see [here](https://arxiv.org/abs/1706.08500), [here](https://arxiv.org/abs/1910.07512)).



## Convergence problems with current algorithms

Unfortunately there are examples of simple functions (such as $f(x,y) = xy$) for which min-max optimization algorithms, such as GDA, may never converge to *any* point (see Figure 1, and our [previous post](https://www.offconvex.org/2020/06/24/equilibrium-min-max/) for a more detailed discussion).

<div style="text-align:center;">
<img src="/assets/GDA_spiral_2.gif" alt="" />
<br>
<b>Figure 1.</b> GDA on $f(x,y) = xy, \, \, \, \, x,y \in [-5,5]$ (the red line is the set of global min-max points). GDA is non-convergent from almost every initial point. 
</div>

<br/>

As for examples relevant to ML, when using GDA to train a GAN on a dataset consisting of points sampled from a mixture of four Gaussians in $\mathbb{R}^2$, we can see that GDA tends to cause the generator to cycle between two or more of these Gaussian modes. We also used GDA to train a GAN on the subset of the MNIST digits which have "0" or "1" as their label, which we refer to as the 0-1 MNIST dataset.  We observed a cycling behavior for this dataset as well: After learning how to generate images of $0$'s, the GAN trained by GDA then forgets how to generate $0$'s for a long time and only generates $1$'s.

<div style="text-align:center;">
<img style="width:400px;" src="/assets/GDA_Gaussian.gif" alt="" />
<img style="width:400px;" src="/assets/GDA_MNIST.gif" alt="" />
<br>
<b>Figure 2.</b> Mode oscillation when GDA is used to train GANs on the four Guassian mixture dataset (left) and the 0-1 MNIST dataset (right).
</div>

<br/>

 In algorithms such as GDA where the discriminator only makes local updates, this cycling behavior can happen for the following reason: Once the discriminator learns to identify one of the modes (say the "$0$" mode), the generator can update $x$ in a way that greatly decreases $f$, by (at least temporarily) "fooling" the discriminator. The generator does this by learning to generate samples from the "$1$" mode which the discriminator has not yet learned to identify, and stops generating samples from the "$0$" mode. However, after many iterations, the discriminator "catches up" to the generator and learns how to identify the "$1$" mode. Since the generator is no longer generating samples from the "$0$" mode, the discriminator may then "forget" how to identify samples from this mode. And this can cause the generator to switch back to generating only the "$0$" mode.



## Our first-order algorithm

To solve the min-max optimization problem, at any point $(x,y)$, we should ideally allow the discriminator to find the global maximum, $\max_z f(x,z)$. However, this may be hard for nonconvex $f$. But we could still let the discriminator run a convergent algorithm (such as gradient ascent) until it reaches a **first-order stationary point**, allowing it to compute an approximation $h$ for the global max function.  (Note that even though $\max_z f(x,z)$ is only a function of $x$, since $h$ is a "local'' approximation it could also depend on the initial point $y$ where we start gradient ascent.) And we also empower the generator to simulate the discriminator's update by running gradient ascent (see [our paper](https://arxiv.org/abs/2006.12376) for discriminators with access to a more general class of first-order algorithms).

> **Idea 1: Use a local approximation to global max**
><br><br>
> Starting at the point $(x,y)$, update $y$ by computing multiple gradient ascent steps for $y$ until a point $w$ is reached where $\|\nabla_y f(x,w)\|$ is close to zero and define $h(x,y) := f(x,w)$.

We would like the generator to minimize $h(\cdot,y)$. To minimize $h$, we would ideally like to update $x$ in the direction $-\nabla_x h$.  However, $h$ may be discontinuous in $x$ (see our [previous post](https://www.offconvex.org/2020/06/24/equilibrium-min-max/) for why this can happen). Moreover, even at points where $h$ is differentiable, computing the gradient of $h$ can take a long time and requires a large amount of memory.

Thus, realistically, we have only zeroth-order access to $h$. A naive approach to minimizing $h$ would be to propose a random update to $x$, for instance an update sampled from a standard Gaussian, and then only accept this update if it causes the value of $h$ to decrease. Unfortunately, this does not lead to fast algorithms as even at points where $h$ is differentiable, in high dimensions, a random Gaussian step will be almost orthogonal to the steepest descent direction $-\nabla_x h(x,y)$, making the progress slow.

Another idea is to propose an update in the direction of the gradient $-\nabla_x f(x,y)$. To see why this may be a reasonable thing to do, notice that once the generator proposes an update $v$ to $x$, the discriminator will only make updates which increase the value of $f$ or, $h(x+v,y)\geq f(x+v,y)$.  And, since $y$ is a first-order stationary point for $f(x,\cdot)$ (because $y$ was computed using gradient ascent in the previous iteration), we also have that $h(x,y) = f(x,y)$. Hence,

$$ f(x+v,y) \leq h(x+v,y) < h(x,y) = f(x,v).$$ 

*This means that decreasing $h$ requires us to decrease $f$ (the converse is not true). So it indeed makes sense to move in the direction $-\nabla_x f(x,y)$!*

While making updates using $-\nabla_x f(x,y)$ may allow the generator to decrease $h$ more quickly than updating in a random direction, it is not always the case that updating in the direction of $-\nabla_x f$ will lead to a decrease in $h$ (and doing so may even lead to an increase in $h$!). Instead, our algorithm has the generator perform a random search by proposing an update in the direction of a batch gradient with mean $-\nabla_x f$, and accepts this move only if the value of $h$ (the local approximation) decreases. The accept-reject step prevents our algorithm from cycling between modes, and using the batch gradient for the random search allows our algorithm to be competitive with prior first-order methods in terms of running time. 

> **Idea 2: Use zeroth-order optimization with batch gradients**
> <br><br>
> Sample a batch gradient $v$ with mean $-\nabla_x f(x,y)$.
><br>
> If $h(x+ v, y) < h(x,y) $ accept the step $x+v$; otherwise reject it.

A final issue, that applies even in the special case of minimization, is that converging to a *local* minimum point does not mean that point is desirable from an application standpoint. The same is true for the more general setting of min-max optimization. To help our algorithm escape undesirable local min-max equilibria, we use a randomized accept-reject rule inspired by [simulated annealing](https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7). Simulated annealing algorithms seek to minimize a function via a randomized search, while gradually decreasing the acceptance probability of this search; in some cases this allows one to reach the global minimum of a nonconvex function (see for instance [this paper](https://arxiv.org/abs/1711.02621)). These three ideas lead us to our algorithm.



> **Our algorithm**
><br><br>
> *Input*: Initial point $(x,y)$, $f: \mathbb{R}^d \times \mathbb{R}^d\rightarrow \mathbb{R}$
><br>
> *Output:* A local min-max equilibrium $(x,y)$
>
> <br> <br>
>
> For $i = 1,2, \ldots$ <br>
> <br>
> *Step 1:* Generate a batch gradient $v$ with mean $-\nabla_x f(x,y)$ and propose the generator update $x+v$.
><br><br>
> *Step 2:* Compute $h(x+v, y) = f(x+v, w)$, by simulating a discriminator update $w$ via gradient ascent on $f(x+v, \cdot)$ starting at $y$.
><br><br>
> *Step 3:*  If $h(x+v, y)$ is less than $h(x,y) = f(x,y)$, accept both updates: $(x,y) = (x+v, w)$. Else, accept both updates with some small probability.



In our paper, we show that our algorithm is guaranteed to converge to a type of local min-max equilibrium in $\mathrm{poly}(\frac{1}{\varepsilon},d, b, L)$ time whenever $f$ is bounded by some $b>0$ and has $L$-Lipschitz gradients. Our algorithm does not require any special starting points, or any additional assumptions on $f$ such as convexity or monotonicity. (See Definition 3.2 and Theorem 3.3 in [our paper](https://arxiv.org/abs/2006.12376))

<div style="text-align:center;">
<img style="width:400px;" src="/assets/GDA_spiral_2.gif" alt="" />
<img style="width:400px;" src="/assets/OurAlgorithm_surface_run1.gif" alt="" />
<br>
<b>Figure 3.</b> GDA (left) and a version of our algorithm (right) on $f(x,y) = xy, \, \, \, \, x,y \in [-5,5]$. While GDA is non-convergent from almost every initial point, our algorithm converges to the set of global min-max points (the red line). To ensure it converges to a (local) equilibrium, our algorithm's generator proposes multiple updates, simulates the discriminator's response, and rejects updates which do not lead to a net decrease in $f$. It only stops if it can't find such an update after many attempts. (To stay inside $[-5,5]\times [-5,5]$ this version of our algorithm uses <i>projected</i> gradients.)
</div>

<br/>

## So, how does our algorithm perform in practice?

When training a GAN on the mixture of four Gaussians dataset, we found that our algorithm avoids the cycling behavior observed in GDA. We ran each algorithm multiple times, and evaluated the results visually. By the 1500'th iteration GDA learned only one mode in 100% of the runs, and tended to cycle between two or more modes. In contrast, our algorithm was able to learn all four modes 68% of the runs, and three modes 26% of the runs.

<div style="text-align:center;">
<img src="/assets/Both_algorithms_Gaussian.gif" alt="" />
<br>
<b>Figure 4.</b> GAN trained using GDA and our algorithm on a four Gaussian mixture dataset. While GDA cycles between the Gaussian modes (red dots), our algorithm learns all four modes.
</div>
<br/>

 When training on the 0-1 MNIST dataset, we found that GDA tends to briefly generate shapes that look like a combination of $0$'s and $1$'s, then switches to generating only $1$'s, and then re-learns how to generate $0$'s. In contrast, our algorithm seems to learn how to generate both $0$'s and $1$'s early on and does not stop generating either digit. We repeated this simulation multiple times for both algorithms, and visually inspected the images at the 1000'th iteration. GANs trained using our algorithm generated both digits by the 1000'th iteration in 86% of the runs, while those trained using GDA only did so in 23% of the runs.

 <div style="text-align:center;">
<img src="/assets/MNIST_bothAlgorithms.gif" alt="" />
<br>
<b>Figure 5.</b> We trained a GAN with GDA and our algorithm on the 0-1 MNIST dataset.  During the first 1000 iterations, GDA "forgets" how to generate $0$'s, while our algorithm learns how to
generate both $0$'s and $1$'s early on and does not stop generating either digit.
</div>

<br/>

While here we have focused on comparing our algorithm to GDA, in [our paper](https://arxiv.org/abs/2006.12376) we include as well a comparison to [Unrolled GANs](https://arxiv.org/abs/1611.02163), which also exhibits cycling between modes. We also present results for CIFAR-10 (see Figures 3 and 7 in our paper), where we compute FID scores to track the progress of our algorithm. See our paper for more details; the code is available on [GitHub](https://github.com/mangoubi/Min-max-optimization-algorithm-for-training-GANs).



## Conclusion

In this post we have shown how to develop a practical and convergent first-order algorithm for training GANs. Our algorithm synthesizes an approximation to the global max function based on first-order algorithms, random search using batch gradients, and simulated annealing. Our simulations show that a version of this algorithm can lead to more stable training of GANs. And yet the amount of memory and time required by each iteration of our algorithm is competitive with GDA. This post, together with the [previous post](https://www.offconvex.org/2020/06/24/equilibrium-min-max/), show that different local approximations to the global max function $\max_z f(x,z)$ can lead to different types of convergent algorithms for min-max optimization. We believe that this idea should be useful in other applications of min-max optimization.

