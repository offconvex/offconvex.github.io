---
layout:     post
title:     Mismatches between Traditional Optimization Analyses and Modern Deep Learning 
date:       2020-10-21 22:00:00
author:     Zhiyuan Li and Sanjeev Arora
visible:    True 
---

You may remember our [previous blog post](http://www.offconvex.org/2020/04/24/ExpLR1/) showing that it is possible to do state-of-the-art deep learning with learning rate that increases exponentially during training.  It was meant to be a dramatic illustration that what we learned in optimization classes and books isn't always a good fit for modern deep learning, specifically, *normalized nets*, which is our term for nets that use any one of popular normalization schemes,e.g. [BatchNorm (BN)](https://arxiv.org/abs/1502.03167), [GroupNorm (GN)](https://arxiv.org/abs/1803.08494), [WeightNorm (WN)](https://arxiv.org/abs/1602.07868). Today's post (based upon [our paper](https://arxiv.org/abs/2010.02916) with Kaifeng Lyu at NeurIPS20)  identifies other surprising incompatibilities between normalized nets and traditional analyses. We hope this will change the way you teach and think about deep learning! 

Before diving into the results, we recall that normalized nets  are typically trained with weight decay (aka $\ell_2$ regularization). Thus the $t$th iteration of Stochastic Gradient Descent (SGD) is:

$$	w_{t+1} \gets (1-\eta_t\lambda)w_t - \eta_t \nabla \mathcal{L}(w_t; \mathcal{B}_t),$$

where $\lambda$ is the weight decay (WD) factor (or $\ell_2$-regularization coefficient),  $\eta_t$ the learning rate, $\mathcal{B}_t$ the batch, and $\nabla \mathcal{L}(w_t,\mathcal{B}_t)$ the batch gradient.

As sketched in our previous blog post, under fairly mild assumptions (namely, fixing the top layer during random initialization ---which empirically does not hurt final accuracy) the loss function for training such normalized nets is *scale invariant*, which means $\mathcal{L}(w _ t; \mathcal{B}_ t)=\mathcal{L}(cw _ t; \mathcal{B} _ t)$, $\forall c>0$.



A consequence of scale invariance is that the $ \nabla _ w \mathcal{L} \vert _ {w = w _ 0} = c \nabla _ w \mathcal{L}\vert _  {w = cw _ 0}$ and $\nabla ^ 2 _ w \mathcal{L} \vert _ {w = w _ 0} = c ^ 2 \nabla ^ 2 _ w \mathcal{L} \vert _  {w = cw _ 0}$, for any $c>0$.










## Some Conventional Wisdoms (CWs)

Now we briefly describe some conventional wisdoms. Needless to say, by the end of this post these will turn out to be very very suspect! Possibly they were OK in earlier days of deep learning, and with shallower nets. 



> CW 1) As we reduce LR to zero, optimization dynamic converges to a deterministic path (Gradient Flow) along which training loss strictly decreases. 

Recall that in traditional explanation of (deterministic) gradient descent, if LR is smaller than roughly the inverse of the smoothness of the loss function, then each step reduces the loss. SGD, being stochastic, has a distribution over possible paths. But very tiny LR can be thought of as full-batch Gradient Descent (GD), which in the limit of infinitesimal step size approaches Gradient Flow (GF).

The above reasoning shows very small LR is guaranteed to decrease the loss at least, as well as any higher LR, can. Of course, in deep learning, we care not only about optimization but also generalization. Here small LR is believed to hurt. 

> CW 2) To achieve the best generalization the LR must be large initially for quite a few epochs. 

This is primarily an empirical finding: using too-small learning rates or too-large batch sizes from the start (all other hyper-parameters being fixed) is known to lead to worse generalization ([Bengio, 2012](https://arxiv.org/pdf/1206.5533.pdf); [Keskar et al., 2017](https://arxiv.org/abs/1609.04836)). 

A popular explanation for this phenomenon is  that the noise in gradient estimation during SGD is beneficial for generalization. (As noted, this noise tends to average out when LR is very small.)  Many authors have suggested that the noise helps becauses it keeps the trajectory away from sharp minima which are believed to generalize worse, although there is some difference of opinion here ([Hochreiter&Schmidhuber, 1997](http://www.bioinf.jku.at/publications/older/3304.pdf); [Keskar et al., 2017](https://arxiv.org/abs/1609.04836); [Li et al., 2018](https://arxiv.org/abs/1712.09913); [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407); [He et al., 2019](https://arxiv.org/pdf/1902.00744.pdf)). [Li et al., 2019](https://arxiv.org/abs/1907.04595) also gave an example (a simple two-layer net) where this observation of worse generalization due to small LR is mathematically proved and also experimentally verified.


>CW 3) Modeling SGD via a Stochastic Differential Equation (SDE) in the continuous-time limit with a fixed Gaussian noise. Namely, think of SGD as a diffusion process that **mixes**  to some Gibbs-like distribution on trained nets. 


This is the usual approach to  formal understanding of CW 2 ([Smith&Le, 2018](https://arxiv.org/abs/1710.06451); [Chaudhari&Soatto, 2018](https://arxiv.org/abs/1710.11029); [Shi et al., 2020](https://arxiv.org/abs/2004.06977)). The idea is that SGD is gradient descent with a noise term, which has a continuous-time approximation as a diffusion process described as 

$$	dW_t = - \eta_t \lambda W_t dt - \eta_t \nabla \mathcal{L}(W_t) dt + \eta_t \Sigma_{W_t}^{1/2} dB_t,$$

where $\sigma_{W_t}$ is the covariance of stochastic gradient $ \nabla \mathcal{L}(w_t; \mathcal{B}_t)$,  and $B_t$ denotes Brownian motion of the appropriate dimension. Several works have adopted this SDE view and given some rigorous analysis of the effect of noise. 

In this story, SGD turns into a geometric random walk in the landscape, which can in principle explore the landscape more thoroughly, for instance by occasionally making loss-increasing steps. While an appealing view, rigorous analysis is difficult because we lack a mathematical description of the loss landscape.  Various papers assume the noise in SDE is isotropic Gaussian, and then derive an expression for the stationary distribution of the random walk in terms of the familiar Gibbs distribution. This view gives intuitively appealing explanation of some deep learning phenomena since the magnitude of noise (which is related to LR and batch size) controls the convergence speed and other properties. For instance it’s well-known that this SDE approximation implies the well-known *linear scaling rule* (Goyal et. al., 2017](https://arxiv.org/pdf/1706.02677.pdf)).

Which raises the question: *does SGD really behave like a diffusion process that mixes in the loss landscape?*

<!--[A few lines explaining for why noise term has this form? e.g., show one step discretization]!-->




## Conventional Wisdom challenged

We now describe the actual discoveries for normalized nets, which show that the above CW's are quite off.

> (Against CW1): Full batch gradient descent $\neq$ gradient flow. 

 It's well known that if LR is smaller than the inverse of the smoothness, then the trajectory of gradient descent will be close to that of gradient flow. But for normalized networks, the loss function is scale-invariant and thus provably non-smooth (i.e., smoothness becomes unbounded)  around the origin ([Li&Arora, 2019](https://arxiv.org/abs/1910.07454)). We show that this non-smoothness issue is very real and makes training unstable and even chaotic for full batch SGD with any nonzero learning rate. This occurs both empirically and provably so with some toy losses. 


<div style="text-align:center;">
<img style="width:60%;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/gd_not_gf.png" />
</div>

**Figure 1.** WD makes GD on scale-invariant loss unstable and chaotic.
(a) Toy model with scale-invariant loss $L(x,y) = \frac{x^2}{x^2+y^2}$  (b)(c) Convergence never truly happens for  ResNet trained on sub-sampled
CIFAR10 containing 1000 images with full-batch GD (without momentum).  ResNet
can easily get to 100% training accuracy but then veers off.  When WD is turned off at epoch 30000 it converges.

 Note that WD plays a crucial role in this effect since without WD the parameter norm increases monotonically 
 ([Arora et al., 2018](https://arxiv.org/abs/1812.03981)) which implies SGD moves away from the origin at all times.


Savvy readers might wonder whether using a smaller LR could fix this issue. Unfortunately, getting close to the origin is unavoidable because once the gradient gets small,  WD will dominate the dynamics and decrease the norm at a geometric rate, causing the gradient to rise again due to the scale invariance! (This happens so long as the gradient gets arbitrarily small, but not actually zero, as is the case in practice.) 


In fact, this is an excellent (and rare) place where early stopping is necessary even for correct optimization of the loss. 

> (Against CW 2) Small LR can generalize equally well as large LR.

This actually was a prediction of the new theoretical analysis we came up with. We ran extensive experiments to test this prediction and found that initial large LR is **not necessary** to match the best performance, even when *all the other hyperparameters are fixed*. See Figure 2.


<div style="text-align:center;">
<img style="width:300px;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/blog_sgd_8000_test_acc.png" />
<img style="width:300px;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/blog_sgd_8000_train_acc.png" />
</div>

**Figure 2**. ResNet trained on CIFAR10 with SGD with normal LR schedule (baseline) as well as a schedule with 100 times smaller initial LR.  The latter matches performance of baseline after one more LR decay!  Note it needs  5000 epochs which is 10x higher! See our paper for details. (Batch size is 128, WD is 0.0005, and LR is divided by 10 for each decay.)

Note the  surprise here is that generalization was not hurt from drastically smaller LR even  *when no other hyperparameter changes*.  It is known empirically as well as rigorously (Lemma 2.4 in [Li&Arora, 2019](https://arxiv.org/abs/1910.07454))  that it is possible to compensate for small LR by other hyperparameter changes. 


>(Against Wisdom 3) Random walk/SDE view of SGD is way off. There is no evidence of mixing as  traditionally understood, at least within normal training times.


Actually the evidence against global mixing exists already via the phenomenon of Stochastic Weight Averaging (SWA) ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)). Along the trajectory of SGD, if  the network parameters from two different epochs are averaged, then the average has test loss lower than either.  Improvement via averaging continues to  work for run times 10X longer  than usual as shown in Figure 3. However, the accuracy improvement doesn't happen for SWA between two solutions obtained from different initialization.  Thus checking whether SWA holds distinguishes between  pairs of solutions drawn from the same trajectory and pairs drawn from different trajectories, which  shows the diffusion process hasn't mixed to stationary distribution within normal training times. (This is not surprising, since the theoretical analysis of mixing does not suggest it happens rapidly at all.) 

<div style="text-align:center;">
<img style="width:300px;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/swa_sgd_test_acc.png" />
<img style="width:300px;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/swa_sgd_dist.png" />
</div>

**Figure 3**. Stochastic Weight Averaging improves the test accuracy of ResNet trained with
SGD on CIFAR10. **Left:** Test accuracy. **Right:** Pairwise distance between parameters from different epochs.

Actually [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407) already noticed the implication that SWA rules out that SGD is a diffusion process which mixes to a unique global equilibrium. They suggested instead that perhaps the trajectory of SGD could be well-approximated by a multivariate Ornstein-Uhlenbeck (OU) process around the *local minimizer* $W^ * $, assuming the loss surface is locally strongly convex. As the corresponding stationary is multi-dimensional Gaussian, $N(W^ *, \Sigma)$, around the local minimizer, $W^ *$, this explains why SWA helps to reduce the training loss.

However, we note that ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407))'s suggestion is also refuted by the fact that we can show $\ell_2$ distance between weights from epochs $T$ and $T+\Delta$ monotonically increases with $\Delta$ for every $T$ (See Figure 3), while $ \mathbf{E} [ \| W_ T-W_ {T+\Delta} \|^2]$ should converge to the constant $2Tr[\Sigma]$ as $T, \Delta \to +\infty$ in the OU process. This suggests that all these weights are correlated, unlike the hypothesized OU process. 
  
## So what's really going on?
  
 We develop a new theory (some parts rigorously proved and others supported by experiments) suggesting that **LR doesn't play the role assumed in most discussions.**  
 
 It's widely believed that LR $\eta$ controls the convergence rate of SGD and affects the generalization via changing the magnitude of noise because LR $\eta$ adjusts the magnitude of gradient update per step. 
 <!--It's also worth noting that for vanilla SGD, changing LR is equivalent to rescaling the loss function. -->
 However, for normalized networks trained with SGD + WD, the effect of LR is more subtle as now it has two roles: (1). the multiplier before the gradient of the loss. (2). the multiplier before WD. Intuitively, one imagines the WD part is  useless since the loss function is scale-invariant, and thus the first role must be more important. But surprisingly, this intuition is completely wrong and it turns out that the second role is way more important than the first one. 
Further analysis shows that a better measure of speed of learning is   $\eta \lambda$, which we call the *intrinsic learning rate* or *intrinsic LR*, denoted $\lambda_e$.

 While previous papers have noticed qualitatively that LR and WD have a close interaction, our ExpLR paper   [Li&Arora, 2019](https://arxiv.org/abs/1910.07454))  gave mathematical proof that *if WD\* LR, i.e., $\lambda\eta$ is fixed, then the effect of changing LR on the dynamics is equivalent to rescaling the initial parameters*.  As far as we can tell, performance of SGD on modern architectures is quite robust to (indeed usually independent of) scale of the initialization, so the effect of changing initial LR while keeping intrinsic LR fixed is also negligible. 
  
 Our paper gives insight into the role of intrinsic LR $\lambda_e$ by giving a new SDE-style analysis of SGD for normalized nets, leading to the following conclusion (which rests in part on experiments):
 
> In normalized nets SGD does indeed lead to rapid mixing, but in **function space** (i.e., input-output behavior of the net). Mixing happens after $O(1/\lambda_e)$ iterations, in contrast to the exponentially slow mixing guaranteed in the parameter space by traditional analysis of diffusion walks. 


To explain the meaning of mixing in function space, let's view SGD (carried out for a fixed number of steps) as a way to sample a trained net from a  distribution over trained nets. Thus the end result of SGD from a fixed initialization can be viewed as a probabilistic classifier whose output on any datapoint is the $K$-dimenstional vector whose $i$th coordinate is the probability of outputting label $i$. (Here $K$ is the total number of labels.) Now if two different initializations both cause SGD to produce classifiers with error $5$ percent on heldout datapoints, then  *a priori* one would imagine that  on a given held-out datapoint the classifier from the first distribution **disagrees**  with the classifier from the second distribution with roughly $2 * 5 =10$ percent probability. (More precisely, $2 * 5 * (1-0.05) = 9.5$ percent.) However, convergence to an equilibrium distribution in function space means that the probability of disagreement is almost $0$, i.e., the distribution is almost the same regardless of the initialization! This is indeed what we experimentally find, to our big surprise. Our theory is built around this new phenomenon.    
 
<div style="text-align:center;">
<img style="width:500px;" src="https://www.cs.princeton.edu/~zl4/small_lr_blog_images/additional_blog_image/conjecture.png" />
</div>
**Figure 4**: A simple 4-layer normalized CNN trained on MNIST with three schedules converge to the same equilibrium after intrinsic LRs become equal at epoch 81. We use Monte Carlo ($500$ trials) to estimate $\ell_1$ distances between distributions. 
 


In the next post, we will explain our new theory and the partial new analysis of SDEs arising from SGD in normalized nets. 
 

  
