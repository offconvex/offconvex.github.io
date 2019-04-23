---
layout: post
title: Is Optimization the Right Language for Deep Learning?
date:  2019-3-11 13:00:00
author: SA + ?? 
visible: False
---

Let's start by acknowledging that the title of this post will appear utterly ridiculous to most experts. All machine learning today involves optimizing some training objective. The objective forces the learner to classify/perform well on the training data, which it surely needs to learn to do.  Furthermore  hundreds of papers at leading machine learning conference  are devoted primarily to various aspects of optimization, and some have actually had big impact. 


In defense let me emphasize that the title is not asking whether optimization is a *useful* concept, but whether it is the right language to ultimately understand deep learning.  This question is driven primarily by the overparametrization phenomenon: an deep learning today often needs to use architectures that are hugely overparametrized for the job at hand. Then the training objective usually has not one but *lots* of global optima. (The paper of Zhang et al. showed empirically that these architectures can fit randomly labeled data, even when we throw in tricks such as regularization, dropout, etc.) Thus the goal is not to find any old global optimum of the objective, but a solution that performs best on unseen/held-out data (i.e., generalizes). And this issue of generalization is where optimization ---specifically, a strict interpretation of optimization as "find me a solution of minimum possible value of the objective, as fast as possible" ---reaches its limits. Because in many settings, value of the objective function may not predict performance on held-out data. 

Of course experts will immediately say: "Wasn't generalization theory invented precisely to address this issue?" For instance this theory shows how to add regularizers to the training objective to ensure the solution generalizes. Or that *early stopping* (stopping before the optimum is reached) or adding noise ---e.g. via stochastic gradient updates---  can be preferable to complete optimization even in simple settings such as regression. 

Personally I have been unable to find any good answers to the deep learning puzzle in current generalization theory  because it is designed to give *post hoc* explanations for why a particular model generalized. We still lack substantial *prescriptions* for how to ensure generalization in overparametrized deep nets, or even overparametrized kernel SVMs for that matter (cite). See my earlier [blog post](http://www.offconvex.org/2017/12/08/generalization1/), which makes a distinction between descriptive and prescriptive methods, and makes the point that generalization theory primarily is of the former type.  Yes, one can add regularizers to the training objective, or do tricks such as dropout or gradient noising (aka playing with batch sizes and learning rates), but they only slightly improve generalization in deep learning. The fundamental mystery is:

> Even vanilla gradient descent is often good at finding models with reasonable generalization. Furthermore, methods to speed up gradient descent (e.g., acceleration methods) can sometimes lead to worse generalization. 

This blog post raises the distinct possibility that much of the magic of deep learning happens along the  gradient descent trajectory and some/most of this magic may not even be formalizable as an explicit single objective that we can attack via optimization. This key role of the trajectory reminds us of the old adage. 

> The journey is more important than the goal. 

Acknowledgements: My views on this topic were originally shaped by the excellent papers from TTI Chicago group regarding the implicit bias of gradient descent (Cite), and were solidified by some recent discoveries that I will sketch below, which involve rigorous analysis of gradient descent in some simple but suggestive settings.  

## Training of Infinitely Wide Deep Nets 

Since overparametrization does not appear to hurt deep nets too much, researchers have wondered what happens if we use a fixed training set e.g., CIFAR10 to train a classic deep net architecture like AlexNet or VGG19 but whose "width" ---namely, number of channels in the convolutional filters, and number of nodes in fully connected internal layers---- is allowed to increase to **infinity**? At first glance this question may seem hopeless and pointless: all the computing in the world is insufficient to train an infinite net, and we theorists already have our hands full trying to figure out finite nets.  But in math/physics there is a tradition of deriving insight into questions by studying them in the infinite limit, and indeed here too the infinite limit becomes easier for theory. 

 Thanks to insights in other recent papers on provable learning by overparametrized deep nets (eg Simon Du, Allen-Zhou, Jason Lee) researchers have realized that the trajectory  of gradient descent ---when nets are initialized in standard ways using Gaussian weights--- acquires a really nice limiting structure. In the maze of infinitely many solutions, gradient descent homes in to a sensible one!
 
 > As the net width goes to infinity: the trajectory of gradient descent approaches the trajectory of gradient descent for some kernel regression problem, where the (fixed) kernel in question is the so-called  *Neural Tangent Kernel* (NTK).
 
  The definition of NTK involves the gradient of the net's output with respect to its parameters. Specifically for two inputs $x_i$ and $x_j$ the kernel inner product   is the inner product of these gradients evaluated at $x_i$, and $x_j$ respectively. This NTK kernel can be efficiently computed, giving us a way to compute the answer of the trained net on any desired input,  *even though training the infinite directly is of course infeasible for us.* (Aside: Please do not confuse these new results with somewhat earlier papers which view infinite nets as Gaussian Processes  ---CITE-- as well as kernels  ---eg Daniely. The object of study there roughly corresponds to an infinite-width deep net where we train only the top layer while freezing the lower layers to a random initialization.) Empirically we find that this infinite net (aka kernel regression with respect to the NTK) yields better performance on CIFAR10 than any previously known kernel ---not counting kernels that were  hand-tuned or designed by training on image data). 
 
## Deep Linear Nets to solve Matrix Completion

Deep Linear Nets, as the name suggests, are deep nets with no nonlinearities. This may well seem like a trivial model: every layer computes a linear transformation, and thus so does the end-to-end model.  What makes it  interesting is that it is overparametrized ---we're using $N$ layers when a single one would have done---and thus it also presents gradient descent with a plethora of solutions to choose from. So does gradient descent end up making an interesting choice? An [earlier blog post by Nadav Cohen](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/) sketched the interesting dynamics of gradient descent, and our discovery that increasing the depth of the net can have an acceleration effect on training. 


In our new paper we study linear nets in context of a well-known optimization problem,  [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion). Here we are given $K$ random entries of a matrix $M$ and we wish to recover the unseen entries. In general this problem has no unique solution, but extensive body of work has shown that if $M$ is low rank or approximately low rank and satisfies some additional technical assumptions (eg *incoherence*) then various algorithms can recover the unseen entries approximately or even exactly. A famous algorithm is based upon nuclear norm minimization: find the matrix that fits the observed entries and has minimum [nuclear norm/trace norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms) among all matrices with this property.  

Based upon experiments and intuition, Neyshabur et al. made an interesting suggestiong. Suppose we try to recover the missing entries by  training a linear net with two layers
 via simple gradient descent/backpropagation. Denoting by $S$ the subset of  all indices $[n]\times [n]$ that was revealed, and  
the revealed entries $\{b_{ij}: (ij) \in S\}$  as labels, train the unknown model  $M_2M_2$ as follows where $e_i$ is the vector with all entries $0$ except for $1$ in the $i$th position:

$$ \textbf{minimize} \sum_{(ij) \in S} (e_i^T(M_2M_1)e_j - b_{ij})^2, $$

The beauty of this phrasing is that "generalization" now corresponds exactly to predicting the missing entries! 

Neyshabur et al. found, surprisingly, that this method does recover the entries quite well, in fact empirically just as well as the above-mentioned nuclear-norm minimization method. In fact they were motivated to make the following conjecture: 

> (Conjecture by Neyshabur et al.; Rough Statement) When used to solve matrix completion as above using a depth $2$ linear net, the solution obtained is exactly the same as the one obtained by the nuclear norm minimization method. 

If this conjecture were true it would imply a very interesting connection between deep learning and a well-known optimization method. But as you may have already guessed, this is too simplistic. In a new paper with Nadav Cohen, Wei Hu and Yuping Luo, we report new experiments suggesting that their conjecture is false. (I hedge by saying "suggest" because some fine print in the conjecture statement makes it pretty hard to refute definitively.) More interesting is our discovery that if we overparametrize even further by increasing the number of layers beyond $2$, even to $3$, then the matrix completion is even better both on synthetic data and on real-life datasets such as [MovieLens](https://grouplens.org/datasets/movielens/100k/). We provide partial analysis for this improved performance of depth $3$ nets by analysing ---surprise surprise!---the trajectory of gradient descent. There appears to be no way to capture what is going on by some simple objective involving matrix Schatten norms (contrary to what we imagined going in to this project). Likewise,  using acceleration tricks such as Adam speed up the optimization (a lot) but slightly hurt generalization. 


## Conclusions/Takeways

The above results involve analysing trajectories of gradient flow (i.e., gradient descent with infinitesimal learning rate). Gradient flow makes parsimonius updates of a special kind that result in very special solutions.
The structure of such solutions would be pretty messy ---if not impossible---to completely capture as a classical optimization problem with a single objective. 

> (Takeaway 1): Different optimization algorithms for deep learning ---SGD, Adam, etc. etc.-- may induce very different trajectories, which may translate into finding solutions with very different properties. Thus the trajectory  may lie at the root of the observed behavior, and not how much/how fast they lower the training objective.


Next, a personal one. 

>(Takeway 2): Sanjeev wishes he'd learnt a few tricks about ODEs/PDEs/Dynamical Systems in college, so he wouldn't have to figure these out 30 years later. 











 

