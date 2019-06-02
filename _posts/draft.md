---
layout: post
title: Is Optimization a Sufficient Language for Understanding Deep Learning?
date:  2019-3-11 13:00:00
author: SA + ?? 
visible: False
---

In this Deep Learning era, machine learning usually boils down to selecting a network architecture, then defining a suitable objective/cost function for the learning task at hand, and finally, optimizing this function using some variant of gradient descent (implemented via backpropagation).  Little wonder that hundreds of ML papers each year are devoted to various aspects of optimization.

Of course, optimization is a necessary component of machine learning, since finding a solution with low objective value is tantamount to forcing the learner to classify/perform well on the training data, which it surely needs to learn to do.
But the point of this blog post is that if our goal is mathematical understanding of deep learning, then  the optimization viewpoint increasingly looks like an insufficient language ---at least in the classical view:

> **(Classic view of optimization)**: *Find me a solution of minimum possible value of the objective, as fast as possible. *

The reason this is insufficient is that deep nets are vastly overparametrized and thus have multiple optima, including those that fit randomly labeled data.  So an important job of the training algorithm is to choose among  optima, and find one that performs well on unseen/held-out data ---i.e., *generalizes.* This important job is left out out of the classical view.

Of course experts will now ask: "Wasn't generalization theory invented precisely for this reason as the "second leg" of machine learning,  where optimization is the first leg?" For instance this theory shows how to add regularizers to the training objective to ensure the solution generalizes. Or that *early stopping* (stopping before the optimum is reached) or even adding noise ---e.g. via stochastic gradient updates---  can be preferable to complete optimization, even in simple settings such as regression. 

However, in practice regularizers  and related tricks such as dropout or gradient noising (eg by playing with batch sizes and learning rates) help generalization a bit. They crtainly can't prevent deep nets from attaining low training objective even on randomly-labeled data (Zhang et al.). In fact current generalization theory is quite dissatisfying because it is designed to give *post hoc* explanations for why a particular model generalized. In keeping with the classic optimization view it is agnostic about *how* the solution was obtained, and thus makes few prescriptions ---apart from recommending some regularization--- for optimization.   (See my earlier [blog post](http://www.offconvex.org/2017/12/08/generalization1/), which explains the distinction between descriptive and prescriptive methods, and  that generalization theory is primarily descriptive.) The fundamental mystery is:

> Even vanilla gradient descent is often good at finding models with reasonable generalization. Furthermore, methods to speed up gradient descent (e.g., acceleration methods) can sometimes lead to worse generalization. 

I have come to feel that much of the magic of deep learning happens due to the precise trajectory of  gradient descent and some/most of this magic may not even be formalizable as an explicit single objective that we can attack via optimization.  This key role of the trajectory reminds us of the old adage. 

> The journey is more important than the goal. 

I will also end with musings about how this may connect to a famous open questions of neuroscience: *Does the brain's workings amount to optimizing a single objective?*

Acknowledgements: My views on this topic were shaped by the excellent papers from TTI Chicago group regarding the implicit bias of gradient descent (Cite), and were solidified by some recent discoveries that I will sketch below, which involve rigorous analysis of gradient descent in some simple but suggestive settings.  

## Training of Infinitely Wide Deep Nets 

Since overparametrization does not appear to hurt deep nets too much, researchers have wondered what happens if we use a fixed training set e.g., CIFAR10 to train a classic deep net architecture like AlexNet or VGG19 but whose "width" ---namely, number of channels in the convolutional filters, and number of nodes in fully connected internal layers---- is allowed to increase to **infinity**? At first glance this question may seem hopeless and pointless: all the computing in the world is insufficient to train an infinite net, and we theorists already have our hands full trying to figure out finite nets.  But in math/physics there is a tradition of deriving insight into questions by studying them in the infinite limit, and indeed here too the infinite limit becomes easier for theory. 

 Thanks to insights in other recent papers on provable learning by overparametrized deep nets (eg Simon Du, Allen-Zhou, Jason Lee) researchers have realized that the trajectory  of gradient descent ---when nets are initialized in standard ways using Gaussian weights--- acquires a really nice limiting structure. In the maze of infinitely many solutions, gradient descent homes in to a sensible one!
 
 > As the net width goes to infinity: the trajectory of gradient descent approaches the trajectory of gradient descent for some kernel regression problem, where the (fixed) kernel in question is the so-called  *Neural Tangent Kernel* (NTK).
 
  The definition of NTK uses the infinite net, and involves the gradient of the net's output with respect to its parameters. Specifically for two inputs $x_i$ and $x_j$ the kernel inner product   is the inner product of these gradients evaluated at $x_i$, and $x_j$ respectively. As the net size increases to infinity this kernel inner product converges, which is why we say the NTK is a fixed kernel. 
  
  Our new paper with Simon Du, Wei Hu, Zhiyuan Li, Russ Salakhutdinov and Ruosang Wang shows that the NTK can be efficiently computed via dynamic programming, giving us a way to efficiently compute the answer of the trained net on any desired input,  *even though training the infinite directly is of course infeasible for us.* (Aside: Please do not confuse these new results with somewhat earlier papers which view infinite nets as Gaussian Processes  ---CITE-- as well as kernels  ---eg Daniely. The object of study there roughly corresponds to an infinite-width deep net where we train only the top layer while freezing the lower layers to a random initialization.) Empirically we find that this infinite net (aka kernel regression with respect to the NTK) yields better performance on CIFAR10 than any previously known kernel ---not counting kernels that were  hand-tuned or designed by training on image data. For instance we can compute the kernel corresponding to a 20-layer convolutional net (CNN) and obtain 78% success rate on CIFAR10.  
 
## Deep Linear Nets to solve Matrix Completion

Deep Linear Nets, as the name suggests, are deep nets with no nonlinearities. This may well seem like a trivial model: every layer computes a linear transformation, and thus so does the end-to-end model.  What makes it  interesting is that it is overparametrized ---we're using $N$ layers when a single one would have done---and thus it also presents gradient descent with a plethora of solutions to choose from. So does gradient descent end up making an interesting choice? An [earlier blog post by Nadav Cohen](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/) sketched the interesting dynamics of gradient descent, and our 2018 discovery that increasing the depth of the net can have an acceleration effect on training. 


In our new paper we study linear nets in context of a well-known optimization problem,  [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion). Here we are given $K$ random entries of a matrix $M$ and we wish to recover the unseen entries. In general this problem has no unique solution, but extensive body of work has shown that if $M$ is low rank or approximately low rank and satisfies some additional technical assumptions (eg *incoherence*) then various algorithms can recover the unseen entries approximately or even exactly. A famous algorithm is based upon nuclear norm minimization: find the matrix that fits the observed entries and has minimum [nuclear norm/trace norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms) among all matrices with this property.  

Based upon experiments and intuition, Neyshabur et al. made an interesting suggestiong. Suppose we try to recover the missing entries by  training a linear net with two layers
 via simple gradient descent/backpropagation. Denoting by $S$ the subset of  all indices $[n]\times [n]$ that was revealed, and  
the revealed entries $\{b_{ij}: (ij) \in S\}$  as labels, train the unknown model  $M_2M_2$ as follows where $e_i$ is the vector with all entries $0$ except for $1$ in the $i$th position:

$$ \textbf{minimize} \sum_{(ij) \in S} (e_i^T(M_2M_1)e_j - b_{ij})^2, $$

The beauty of this phrasing is that "generalization" now corresponds exactly to doing well at predicting a random missing entry! 

Neyshabur et al. found, surprisingly, that this method does recover the entries quite well, in fact empirically just as well as the above-mentioned nuclear-norm minimization method. In fact they were motivated to make the following conjecture: 

> (Conjecture by Gunasekar et al.; Rough Statement) When used to solve matrix completion as above using a depth $2$ linear net, the solution obtained is exactly the  one obtained by the nuclear norm minimization method. 

If this conjecture were true it would imply a very interesting connection between deep learning and a well-known optimization method. But as you may have already guessed, this is too simplistic. In a new paper with Nadav Cohen, Wei Hu and Yuping Luo, we report new experiments suggesting that the conjecture is false. (I hedge by saying "suggest" because some fine print in the conjecture statement makes it pretty hard to refute definitively.) More interesting, we find that if we overparametrize the problem by further increasing the number of layers from $2$ to $3$ then this solves matrix completion even better, both on synthetic data and on real-life datasets such as [MovieLens](https://grouplens.org/datasets/movielens/100k/). We provide partial analysis for this improved performance of depth $3$ nets by analysing ---surprise surprise!---the trajectory of gradient descent. We show that roughly speaking depth $3$ networks build up the solution by preferentially evolving a singular value $\sigma$ at the rate $\sigma^{4/3}$. (Furthermore, if the depth is $N$ then the preference is like $\sigma^{2-2/N}$.) Stated this way, the effect of gradient descent may appear to be capturable by some other matrix norm, but we dash this hope by also present results that suggest there may be no way to capture the full range of effects by some simple objective involving matrix Schatten norms (contrary to what we imagined going in to this project). 

It is also interesting to note that we empirically find that Adam, the celebrated  acceleration method for deep learning, empirically speeds up the optimization a lot, but slightly hurts generalization. 


## Conclusions/Takeways

The above results involve analysing trajectories of gradient flow (i.e., gradient descent with infinitesimal learning rate). Gradient flow makes parsimonius updates of a special kind that result in very special solutions.
The structure of such solutions would be pretty messy ---if not impossible---to completely capture as a classical optimization problem with a single objective. 

> (Takeaway 1): Different optimization algorithms for deep learning ---SGD, Adam, etc. etc.-- may induce very different trajectories, which may translate into finding solutions with very different properties. Thus the trajectory  may lie at the root of the observed behavior, and not how much/how fast they lower the training objective.

> (Takewaway 2) Classic optimization work often takes the "landscape view" where one worries about how stationary points, gradient norms, Hessian norms, smoothness etc. behave in the full loss landscape. For deep learning we need a new vocabulary for reasoning about trajectories, and mathematics explaining what kinds of trajectories arise during gradient-based training. Note that trajectories depend on initialization, so there is in principle a continuum of trajectories to think about. 



>(Takeway 3): Sanjeev wishes he'd learnt a few tricks about ODEs/PDEs/Dynamical Systems in college, so he were better equipped for reasoning about trajectories. 











 

