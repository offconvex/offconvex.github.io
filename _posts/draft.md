---
layout: post
title: Is Optimization a Sufficient Language for Understanding Deep Learning?
date:  2019-3-11 13:00:00
author: SA + ?? 
visible: False
---

In this Deep Learning era, machine learning usually boils down to defining a suitable objective/cost function for the learning task at hand, and then optimizing this function using some variant of gradient descent (implemented via backpropagation).  Little wonder that hundreds of ML papers each year are devoted to various aspects of optimization. Today I will suggest that if our goal is mathematical understanding of deep learning, then  the optimization viewpoint is potentially insufficient ---at least in the conventional view:

> **Conventional View (CV) of Optimization**: Find me a solution of minimum possible value of the objective, as fast as possible. 

Note that *a priori* it is not obvious if all learning should involve optimizing a single objective. Whether or not this is true for the working of the brain is a longstanding open question in neuroscience. Brain components appear to have been repurposed/cobbled together through various accidents of evolution and the whole assemblage may or may not boil down to optimization of an objective. See [this survey by Marblestone et al](https://arxiv.org/pdf/1606.03813.pdf)). 

I am suggesting that deep learning algorithms also have important properties that are not always reflected in the objective value. Current deep nets, being vastly overparametrized, have multiple optima. (They are trained until the objective is almost zero.)  The training is said to succeed if the optimum (or near-optimum) model it found also performs well on unseen/held-out data ---i.e., *generalizes.* The value of the objective training says nothing about generalization. 

Of course experts will now ask: "Wasn't generalization theory invented precisely for this reason as the "second leg" of machine learning,  where optimization is the first leg?" For instance this theory shows how to add regularizers to the training objective to ensure the solution generalizes. Or that *early stopping* (i.e., stopping before reaching the optimum) or even adding noise to the gradient (e.g. by playing with batch sizes and learning rates) can be preferable to perfect optimization, even in simple settings such as regression. 

However, in practice explicit regularizers  and noising tricks can't prevent deep nets from attaining low training objective even on randomly-labeled data, which obviously don't generalize; see [Zhang et al.](https://arxiv.org/abs/1611.03530). Current generalization theory is designed to give *post hoc* explanations for why a particular model generalized. It is agnostic about *how* the solution was obtained, and thus makes few prescriptions ---apart from recommending some regularization--- for optimization.   (See my earlier [blog post](http://www.offconvex.org/2017/12/08/generalization1/), which explains the distinction between descriptive and prescriptive methods, and  that generalization theory is primarily descriptive.) The fundamental mystery is:

> Even vanilla gradient descent (GD) is good at finding models with reasonable generalization. Furthermore, methods to speed up gradient descent (e.g., acceleration or adaptive regularization) can sometimes lead to worse generalization. 

In other words, GD has an innate bias towards finding solutions with good generalization. Magic happens along the GD trajectory and is not captured in the objective value per se. We're reminded of the old adage. 

> The journey matters more than the destination. 

Below to illustrate this viewpoint I describe new papers with rigorous analysis of gradient descent in two simple but suggestive settings. I am barely sketching the results and hope that more details will appear in future blog posts.

Acknowledgements: My views on this topic were initially shaped by the excellent papers from TTI Chicago group regarding the implicit bias of gradient descent ([Behnam Neyshabur's thesis](https://arxiv.org/pdf/1709.01953.pdf) is a good starting point), and of course by my various coauthors. 

## Computing with Infinitely Wide Deep Nets 

Since overparametrization does not appear to hurt deep nets too much, researchers have wondered what happens in the infinite limit of overparametrization: use a fixed training set such as CIFAR10 to train a classic deep net architecture like AlexNet or VGG19 whose "width" ---namely, number of channels in the convolutional filters, and number of nodes in fully connected internal layers---- is allowed to increase to **infinity**. Perhaps such questions seem hopeless and pointless: all the computing in the world is insufficient to train an infinite net, and we theorists already have our hands full trying to figure out finite nets.  But sometimes in math/physics one can derive insight into questions by studying them in the infinite limit. Note that initialization (using sufficiently small Gaussian weights) and training makes sense for any finite width, no matter how large. We assume $\ell_2$ loss at the output. Now the number of optima for training objective shoots to $\infty$ as well! 

 Thanks to insights in recent papers on provable learning by overparametrized deep nets (eg Simon Du, Allen-Zhou, Jason Lee) researchers have realized that as width $\rightarrow \infty$ the a nice limiting structure emerges:
 
 > As width $\rightarrow \infty$, trajectory approaches the trajectory of GD for a kernel regression problem, where the (fixed) kernel in question is the so-called  *Neural Tangent Kernel* (NTK). (For convolutional nets the kernel is *Convolutional NTK or CNTK.* )
 
  The definition of this fixed kernel uses the infinite net. For  two inputs $x_i$ and $x_j$ the kernel inner product  $K(x_i, x_j)$  is the inner product of the gradient of the output with respect to the input, evaluated at $x_i$, and $x_j$ respectively. As the net size increases to infinity this kernel inner product converges to a limiting value (there is a technicality about how to define the limit, and the series of new papers have improved the formal statement here). 
  
  Our [new paper](https://arxiv.org/abs/1904.11955) with Simon Du, Wei Hu, Zhiyuan Li, Russ Salakhutdinov and Ruosang Wang shows that the CNTK can be efficiently computed via dynamic programming, giving us a way to efficiently compute the answer of the trained net on any desired input,  *even though training the infinite net directly is of course infeasible.* (Aside: Please do not confuse these new results with somewhat earlier papers which view infinite nets as Gaussian Processes  ---CITE-- as well as kernels  ---eg Daniely. The object of study there roughly corresponds to an infinite-width deep net where we train only the top layer while freezing the lower layers to a random initialization.) Empirically we find that this infinite net (aka kernel regression with respect to the NTK) yields better performance on CIFAR10 than any previously known kernel ---not counting kernels that were  hand-tuned or designed by training on image data. For instance we can compute the kernel corresponding to a 20-layer convolutional net (CNN) and obtain 78% success rate on CIFAR10. 
  
  
## Deep Matrix Factorization to solve Matrix Completion

 [Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion), motivated by design of recommender systems, has been well-studied for over a decade: given $K$ random entries of an unknown matrix we wish to recover the unseen entries. This has no unique solution in general. But if the unknown matrix is low rank or approximately low rank and satisfies some additional technical assumptions (eg *incoherence*) then various algorithms can recover the unseen entries approximately or even exactly. A famous algorithm based upon [nuclear/trace norm](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms)  minimization goes as follows: find matrix that fits all the known observations and has minimum nuclear norm. In line with the Conventional View we can phrase it as optimizing a single objective as follows where $S$ is the subset of indices of revealed entries, $|M|_{*}$ denotes nuclear norm and $\lambda$ is a multiplier.
  
$$\textbf{minimize} $\sum_{(ij) \in S} (M_{ij} - b_{ij})^2 + \lambda |M|_{*}.$$ 

[Gunasekar et al.](http://papers.nips.cc/paper/7195-implicit-regularization-in-matrix-factorization) made an interesting suggestion: suppose we forget entirely about the nuclear norm and try to recover the missing entries by  training a linear net with two layers via simple gradient descent/backpropagation. This linear net is just a multiplication of two $n\times n $ matrices (to learn about linear deep nets see this [earlier blog post by Nadav Cohen](http://www.offconvex.org/2018/03/02/acceleration-overparameterization/)) so we obtain the following  where $e_i$ is the vector with all entries $0$ except for $1$ in the $i$th position:

$$ \textbf{minimize} \sum_{(ij) \in S} (e_i^T(M_2M_1)e_j - b_{ij})^2, $$

The "data" now correspond to indices $(i, j) \in S$  and since $S$ was chosen randomly among all entries,  "generalization" corresponds exactly to doing well at predicting the remaining entries. Empirically, soving matrix completion this way via deep learning  (i.e., gradient descent to solve for $M_1, M_2$) works pretty similarly  to the classic algorithm, leading to the following conjecture: 

> (Conjecture by Gunasekar et al.; Rough Statement) When used to solve matrix completion as above using a depth $2$ linear net, the solution obtained is exactly the  one obtained by the nuclear norm minimization method. 

If this conjecture were true it would imply that the implicit regularization effect of gradient descent is captured exactly by the nuclear norm. But as you may have already guessed, this is too simplistic. In a new paper with Nadav Cohen, Wei Hu and Yuping Luo, we report new experiments suggesting that the above conjecture is false. (I hedge by saying "suggest" because some fine print in the conjecture statement makes it pretty hard to refute definitively.) More interesting, we find that if we overparametrize the problem by further increasing the number of layers from $2$ to $3$ ---which we call Deep Matrix Factorization---then this empirically solves matrix completion even better than nuclear norm does. (Note that we're working in the regime where $S$ is slightly smaller than what it needs to be for nuclear norm algorithm to work well. Inductive bias is most important precisely in such data-poor settings!) We provide partial analysis for this improved performance of depth $3$ nets by analysing ---surprise surprise!---the trajectory of gradient descent and showing how it biases toward finding solutions of low rank. Furthermore we show results suggesting that this bias toward low rank  cannot be captured by nuclear norm or any obvious quasi-Schatten norm.

NB: empirically we find that Adam, the celebrated  acceleration method for deep learning, empirically speeds up the optimization a lot here as well, but slightly hurts generalization. This also relates to what I said above about the Conventional View. 

## Conclusions/Takeways


(1): Different optimization algorithms for deep learning ---SGD, Adam, AdaGrad etc. etc.-- may induce very different trajectories, which may translate into finding solutions with very different properties. The trajectory  may explain a lot many properties of the solution, and not the value of the objective, or how fast the algorithm runs. 

(2) Conventional View of ptimization leads naturally to the "landscape view" where one worries about stationary points, gradient norms, Hessian norms, smoothness etc. For deep learning we need a new vocabulary (and mathematics) to reason about trajectories arising during gradient-based training. Caution: trajectories depend on initialization! 

(3): I wish I had learnt a few tricks about ODEs/PDEs/Dynamical Systems in college, to be in better shape to reason about trajectories!











 

