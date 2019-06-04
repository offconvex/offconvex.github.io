---
layout: post
title: Is Optimization a Sufficient Language for Understanding Deep Learning?
date:  2019-06-03 10:00:00
author: Sanjeev Arora
visible: True
---

In this Deep Learning era, machine learning usually boils down to defining a suitable objective/cost function for the learning task at hand, and then optimizing this function using some variant of gradient descent (implemented via backpropagation).  Little wonder that hundreds of ML papers each year are devoted to various aspects of optimization. Today I will suggest that if our goal is mathematical understanding of deep learning, then  the optimization viewpoint is potentially insufficient ---at least in the conventional view:

> **Conventional View (CV) of Optimization**: Find a solution of minimum possible value of the objective, as fast as possible. 

Note that *a priori* it is not obvious if all learning should involve optimizing a single objective. Whether or not this is true for the learning in the brain is a longstanding open question in neuroscience. Brain components appear to have been repurposed/cobbled together through various accidents of evolution and the whole assemblage may or may not boil down to optimization of an objective. See [this survey by Marblestone et al](https://arxiv.org/pdf/1606.03813.pdf). 

I am suggesting that deep learning algorithms also have important properties that are not always reflected in the objective value. Current deep nets, being vastly overparametrized, have multiple optima. They are trained until the objective is almost zero (i.e., close to optimality) and training is said to succeed if the optimum (or near-optimum) model thus found also performs well on unseen/held-out data ---i.e., *generalizes.* The catch here is that the value of the objective may imply nothing about generalization (see [Zhang et al.](https://arxiv.org/abs/1611.03530)). 

Of course experts will now ask: "Wasn't generalization theory invented precisely for this reason as the "second leg" of machine learning,  where optimization is the first leg?" For instance this theory shows how to add regularizers to the training objective to ensure the solution generalizes. Or that *early stopping* (i.e., stopping before reaching the optimum) or even adding noise to the gradient (e.g. by playing with batch sizes and learning rates) can be preferable to perfect optimization, even in simple settings such as regression. 

However, in practice explicit regularizers  and noising tricks can't prevent deep nets from attaining low training objective even on data with random labels; see [Zhang et al.](https://arxiv.org/abs/1611.03530). Current generalization theory is designed to give *post hoc* explanations for why a particular model generalized. It is agnostic about *how* the solution was obtained, and thus makes few prescriptions ---apart from recommending some regularization--- for optimization.   (See my earlier [blog post](http://www.offconvex.org/2017/12/08/generalization1/), which explains the distinction between descriptive and prescriptive methods, and  that generalization theory is primarily descriptive.) The fundamental mystery is:

> Even vanilla gradient descent (GD) is good at finding models with reasonable generalization. Furthermore, methods to speed up gradient descent (e.g., acceleration or adaptive regularization) can sometimes lead to worse generalization. 

In other words, GD has an innate bias towards finding solutions with good generalization. Magic happens along the GD trajectory and is not captured in the objective value per se. We're reminded of the old adage. 

> The journey matters more than the destination. 

I will illustrate this viewpoint by sketching new  rigorous analyses of gradient descent in two simple but suggestive settings. I  hope more  detailed writeups will appear in future blog posts.

Acknowledgements: My views on this topic were initially shaped by the excellent papers from TTI Chicago group regarding the implicit bias of gradient descent ([Behnam Neyshabur's thesis](https://arxiv.org/pdf/1709.01953.pdf) is a good starting point), and then of course by  various coauthors. 

## Computing with Infinitely Wide Deep Nets 

Since overparametrization does not appear to hurt deep nets too much, researchers have wondered what happens in the infinite limit of overparametrization: use a fixed training set such as CIFAR10 to train a classic deep net architecture like AlexNet or VGG19 whose "width" ---namely, number of channels in the convolutional filters, and number of nodes in fully connected internal layers---- is allowed to increase to **infinity**. Note that initialization (using sufficiently small Gaussian weights) and training makes sense for any finite width, no matter how large. We assume $\ell_2$ loss at the output. 

Understandably, such questions can seem hopeless and pointless: all the computing in the world is insufficient to train an infinite net, and we theorists already have our hands full trying to figure out finite nets.  But sometimes in math/physics one can derive insight into questions by studying them in the infinite limit.  Here where an infinite net is training on a finite dataset like CIFAR10, the number of optima is infinite and we are trying to understand what GD does. 

 Thanks to insights in recent papers on provable learning by overparametrized deep nets (some of the key papers are: [Allen-Zhu et al 1](https://arxiv.org/abs/1811.04918), [Allen-Zhu et al 2](https://arxiv.org/abs/1811.03962) [Du et al](https://arxiv.org/abs/1811.03804), [Zou et al](https://arxiv.org/abs/1811.08888)) researchers have realized that a nice limiting structure emerges:
 
 > As width $\rightarrow \infty$, trajectory approaches the trajectory of GD for a kernel regression problem, where the (fixed) kernel in question is the so-called  *Neural Tangent Kernel* (NTK). (For convolutional nets the kernel is *Convolutional NTK or CNTK.* )
 
 The kernel was identified and named by [Jacot et al.](https://arxiv.org/abs/1806.07572), and also implicit in some of the above-mentioned papers on overparametrized nets, e.g. [Du et al](https://arxiv.org/abs/1810.02054).
 
  The definition of this fixed kernel uses the infinite net at its random initialization. For  two inputs $x_i$ and $x_j$ the kernel inner product  $K(x_i, x_j)$  is the inner product of the gradient $\nabla_x$ of the output with respect to the input, evaluated at $x=x_i$, and $x= x_j$ respectively. As the net size increases to infinity this kernel inner product can be shown to converge to a limiting value (there is a technicality about how to define the limit, and the series of new papers have improved the formal statement here; eg [Yang2019](https://arxiv.org/abs/1902.04760) and our paper below.). 
  
  Our [new paper with Simon Du, Wei Hu, Zhiyuan Li, Russ Salakhutdinov and Ruosang Wang](https://arxiv.org/abs/1904.11955) shows that the CNTK can be efficiently computed via dynamic programming, giving us a way to efficiently compute the answer of the trained net for any desired input,  *even though training the infinite net directly is of course computationally infeasible.* (Aside: Please do not confuse these new results with some earlier papers which view infinite nets as kernels or Gaussian Processes ---see citations/discussion in our paper---  since they correspond to training only the top layer while freezing the lower layers to a random initialization.) Empirically we find that this infinite net (aka kernel regression with respect to the NTK) yields better performance on CIFAR10 than any previously known kernel ---not counting kernels that were  hand-tuned or designed by training on image data. For instance we can compute the kernel corresponding to a 10-layer convolutional net (CNN) and obtain 77.4% success rate on CIFAR10. 
  
  
## Deep Matrix Factorization for solving Matrix Completion

 [Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion), motivated by design of recommender systems, is well-studied for over a decade: given $K$ random entries of an unknown matrix, we wish to recover the unseen entries. Solution is not unique in general. But if the unknown matrix is low rank or approximately low rank and satisfies some additional technical assumptions (eg *incoherence*) then various algorithms can recover the unseen entries approximately or even exactly. A famous algorithm  based upon [nuclear/trace norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms)  minimization as follows: find matrix that fits all the known observations and has minimum nuclear norm. (Note that nuclear norm is a convex relaxation of rank.) It is also possible to rephrase this as a single objective in the form required by the Conventional View as follows where $S$ is the subset of indices of revealed entries,  $\lambda$ is a multiplier:
  
$$\textbf{minimize} \sum_{(ij) \in S} (M_{ij} - b_{ij})^2 + \lambda |M|_{*}.$$ 

In case you didn't know about nuclear norms, you will like the interesting suggestion made by [Gunasekar et al.](http://papers.nips.cc/paper/7195-implicit-regularization-in-matrix-factorization): let us just forget about the nuclear norm penalty term  altogether. Instead try to recover the missing entries by  simply training (via simple gradient descent/backpropagation) a linear net with two layers on the first term in the loss. This linear net is just a multiplication of two $n\times n $ matrices (you can read about linear deep nets in this [earlier blog post by Nadav Cohen](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/)) so we obtain the following  where $e_i$ is the vector with all entries $0$ except for $1$ in the $i$th position:

$$ \textbf{minimize} \sum_{(ij) \in S} (e_i^T(M_2M_1)e_j - b_{ij})^2, $$

The "data" now corresponds to indices $(i, j) \in S$, and the training loss captures how well the end-to-end model $M_2M_1$ fits the revealed entries.  Since $S$ was chosen randomly among all entries,  "generalization" corresponds exactly to doing well at predicting the remaining entries. Empirically, soving matrix completion this way via deep learning  (i.e., gradient descent to solve for $M_1, M_2$, and entirely forgetting about ensuring low rank) works as well as the classic algorithm, leading to the following conjecture, which if true would imply that the implicit regularization effect of gradient descent in this case is captured exactly by the nuclear norm. 

> (Conjecture by Gunasekar et al.; Rough Statement) When solving matrix completion as above using a depth-$2$ linear net, the solution obtained is exactly the  one obtained by the nuclear norm minimization method. 

But as you may have already guessed, this turns out to be too simplistic. In [a new paper with Nadav Cohen, Wei Hu and Yuping Luo](https://arxiv.org/abs/1905.13655), we report new experiments suggesting that the above conjecture is false. (I hedge by saying "suggest" because some fine print in the conjecture statement makes it pretty hard to refute definitively.) More interesting, we find that if we overparametrize the problem by further increasing the number of layers from two to $3$ or even higher ---which we call Deep Matrix Factorization---then this empirically solves matrix completion even better than nuclear norm minimization. (Note that we're working in the regime where $S$ is slightly smaller than what it needs to be for nuclear norm algorithm to exactly recover the matrix. Inductive bias is most important precisely in such data-poor settings!) We provide partial analysis for this improved performance of depth $N$ nets by analysing ---surprise surprise!---the trajectory of gradient descent and showing how it biases strongly toward finding solutions of low rank, and this bias is stronger than simple nuclear norm. Furthermore our analysis suggests that this bias toward low rank  cannot be captured by nuclear norm or any obvious Schatten quasi-norm of the end-to-end matrix. 

NB: Empirically we find that Adam, the celebrated  acceleration method for deep learning, speeds up optimization a lot here as well, but slightly hurts generalization. This relates to what I said above about the  Conventional View being insufficient to capture generalization.

## Conclusions/Takeways

Though the above settings are simple, they suggest that to understand deep learning we have to go beyond the Conventional View of optimization, which focuses only on the value of the objective and the rate of convergence.

(1): Different optimization strategies ---GD, SGD, Adam, AdaGrad etc. ----lead to different learning algorithms. They induce different trajectories, which may lead to solutions with different generalization properties. 

(2) We need to develop a new vocabulary (and mathematics) to reason about trajectories. This goes beyond the usual "landscape view" of stationary points, gradient norms, Hessian norms, smoothness etc. Caution: trajectories depend on initialization! 

(3): I wish I had learnt a few tricks about ODEs/PDEs/Dynamical Systems/Lagrangians in college, to be in better shape to reason about trajectories!











 

