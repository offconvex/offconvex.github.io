---
layout: post
title: Unsupervised learning, one notion or many?
date:  2017-06-27 04:00:00
author: Sanjeev Arora and Andrej Risteski
visible: True
---
---
layout: post
title: Unsupervised learning, one notion or many?
date:  2017-03-20 13:00:00
author: Sanjeev Arora
visible: False
---

*Unsupervised learning*, as the name suggests, is the science of learning from unlabeled data. A look at the [wikipedia page](https://en.wikipedia.org/wiki/Unsupervised_learning) shows that this term has many interpretations: 

**(Task A)**  *Learning a distribution from samples.* (Examples: gaussian mixtures, topic models, variational autoencoders,..)

**(Task B)** *Understanding latent structure in the data.* This is not the same as (a); for example  principal component analysis, clustering, manifold learning etc. identify latent structure but don't learn a distribution per se.

**(Task C)** *Feature Learning.* Learn a mapping from *datapoint* $\rightarrow$ *feature vector* such that classification tasks are easier to carry out on feature vectors rather than datapoints. For example, unsupervised feature learning could help lower the amount of *labeled* samples needed for learning a classifier, or  be useful for [*domain adaptation*](https://en.wikipedia.org/wiki/Domain_adaptation).

Task B is often a subcase of Task C, as structure in data is found by humans pouring over the representation of data to see if some intuitive property is satisfied, which can be often phrased as a classification task. 



This post explains the relationship between  Tasks A and C, and why they get mixed up in students' mind. We hope  there is also some food for thought here for experts, namely, our discussion about the fragility of the usual "perplexity" definition of unsupervised learning. It explains why Task A doesn't in practice lead to good enough solution for Task C. 


## The common theme: high level representations. 	

If $x$ is a datapoint, each of these methods seeks to map it to a new "high level" representation $h$ that captures its "essence."
This is why it helps to have access to $h$ when performing machine learning tasks on $x$ (e.g. classification). 
The difficulty of course is that  "high-level representation" is not uniquely defined. For example, $x$ may be an image, and $h$ may  contain the information that it contains a person and a dog. But another  $h$ may say that it shows a poodle and a person wearing pyjamas standing on the beach. This nonuniqueness seems inherent. 
 

Unsupervised learning tries to learn high-level representation using unlabeled data.
Each method make an implicit assumption about how the hidden $h$ relates to the visible $x$. For example, in k-means clustering the hidden $h$ consists of  labeling the datapoint with the index of the cluster it belongs to. 
Clearly, such a simple clustering-based representation has rather limited  expressive power since it groups datapoints into disjoint classes: this limits its application for complicated settings. For example, if one clusters images according to the labels "human", "animal" "plant" etc., then which cluster should contain an image showing a man and a dog standing in front of a tree? 


The search for a descriptive language for talking about the possible relationships of representations and data leads us naturally to Bayesian models. (Note that these are viewed with some skepticism in  machine learning theory -- compared to assumptionless models like PAC learning, online learning, etc. -- but we do not know of another suitable vocabulary in this setting.) 

## A Bayesian view 

Bayesian approaches  capture the relationship between the "high level"  representation $h$ and the datapoint $x$ by postulating a *joint distribution*  $p_{\theta}(x, h)$ of the data $x$ and representation $h$, such that $p_{\theta}(h)$ and the posterior $p_{\theta}(x \mid h)$ have a simple form as a function of the parameters $\theta$. These are also called *latent variable* probabilistic models, since $h$ is a latent (hidden) variable.

The standard goal in distribution learning is to find the $\theta$ that "best explains" the data (what we called Task (A)) above). This is formalized using maximum-likelihood estimation going back to Fisher (~1910-1920): find the $\theta$ that maximizes the *log probability* of the training data. Mathematically, indexing the samples with $t$, we can write this as
$$  \max_{\theta} \sum_{t} \log p_{\theta}(x_t)  \qquad (1) $$

where
$$p_{\theta}(x_t) = \sum_{h_t}p_{\theta}(x_t, h_t). $$

(Note that $\sum_{t} \log p_{\theta}(x_t)$ is also the empirical estimate of the *cross-entropy* 
$E_{x}[\log p_{\theta}(x)]$ of the distribution $p_{\theta}$, where $x$ is distributed according to $p^*$, the true distribution of the data. Thus the above method looks for the distribution with best cross-entropy on the empirical data, which is also log of the [*perplexity*](https://en.wikipedia.org/wiki/Perplexity) of $p_{\theta}$.) 

 In the limit of $t \to ∞$, this estimator is *consistent* (converges in probability to the ground-truth value) and *efficient* (has lowest asymptotic mean-square-error among all consistent estimators). See the [Wikipedia page](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). (Aside: maximum likelihood estimation is often NP-hard, which is one of the reasons for the renaissance of the method-of-moments and tensor decomposition algorithms in learning latent variable models, which [Rong  wrote about some time ago](http://www.offconvex.org/2015/12/17/tensor-decompositions/).)  

### Toward task C: Representations arise from the posterior distribution

Simply learning the distribution $p_{\theta}(x, h)$ does not yield a representation *per se.* To get a distribution of $x$, we need access to the posterior $p_{\theta}(h \mid x)$: then a sample from this posterior can be used as a "representation" of a data-point $x$. (Aside: Sometimes, in settings when $p_{\theta}(h \mid x)$ has a simple description, this description can be viewed as the representation of $x$.)
 
Thus solving Task C requires learning distribution parameters $\theta$  *and* figuring out how to efficiently sample from the posterior distribution. 

Note that the sampling problems for the posterior can be \#-P hard for very simple families. The reason is that by Bayes law, $p_{\theta}(h \mid x) = \frac{p_{\theta}(h) p_{\theta}(x \mid h)}{p_{\theta}(x)}$. Even if the numerator is  easy to calculate, as is the case for simple families, the   $p_{\theta}(x)$ involves a big summation (or integral) and is often hard to calculate. 

Note that the  max-likelihood parameter estimation (Task A) and approximating the posterior distributions $p(h|x)$ (Task C) can have radically different complexities: Sometimes A is easy but C is NP-hard (like topic modeling with "nice" topic-word matrices, but short documents, see also [Bresler 2015](https://arxiv.org/abs/1411.6156)); or vice versa (like topic modeling with long documents, but worst-case chosen topic matrices [Arora et al. 2011](https://arxiv.org/abs/1111.0952)) 

Of course, one may hope (as usual) that computational complexity is a worst-case notion and may not apply in practice. But there is a bigger issue with this setup, having to do with accuracy.

## Main Issue: Accuracy (why the above reasoning is fragile)

The above description assumes that the parametric model $p_{\theta}(x, h)$ for the data was *exact* whereas one imagines it is only *approximate* (i.e., suffers from modeling error). Furthermore, computational difficulties may restrict us to use approximately correct inference  even if the model were exact. So in practice, we may only have an *approximation* $q(h|x)$ to 
the posterior distribution  $p_{\theta}(h \mid x)$. (Below we describe a popular methods to compute such approximations.) 

>  *How good of an approximation* to the true posterior do we need?

Recall, we are trying to answer this question through the lens of Task C, solving some classification task. We take the following point of view: 

> For $t=1, 2,\ldots,$ nature picked some $(h_t, x_t)$ from the joint distribution  and presented us $x_t$. The true label $y_t$ of $x_t$ is $\mathcal{C}(h_t)$ where  $\mathcal{C}$ is an unknown classifier. Our goal is classify according to these labels.

 To simplify notation, assume the output of $\mathcal{C}$ is binary. If we wish to use 
 $q(h \mid x)$ as a surrogate for the true posterior $p_{\theta}(h \mid x)$, we need to have 
 $$\Pr_{x_t, h_t \sim q(\cdot \mid x_t)} [\mathcal{C}(h_t) \neq y_t] \mbox{  is small as well.} $$ 

How close must $q(h \mid x)$ and $p(h \mid x)$ be to let us conclude this? We will use KL divergence as "distance" between the distributions, for reasons that will become apparent in the following section. We claim the following: 

> The classification error on representations obtained using $q(h_t \mid x_t)$ is less than $\epsilon$ if $KL(q(h_t \mid x_t) || p(h_t \mid x_t)) \leq 2\epsilon^2.$

Here's a proof sketch.   The natural distance these two distributions $q(h \mid x)$ and $p(h \mid x)$ with respect to accuracy of classification tasks is *total variation (TV)* distance. Indeed, if the TV distance between $q(h\mid x)$ and $p(h \mid x)$ is bounded by $\epsilon$, this implies that for any event $\Omega$, $$\left|\Pr_{h_t \sim p(\cdot|x_t)}[\Omega] - \Pr_{h_t \sim q(\cdot \mid x_t)}[\Omega]\right| \leq \epsilon .$$ Instantiating this inequality with the event $\Omega = $ { $\mathcal{C}(h_t) \neq y_t$ }, we get $$\left|\Pr_{h_t \sim p(\cdot \mid x_t)}[\mathcal{C}(h_t) \neq y_t] - \Pr_{h_t \sim q(\cdot \mid x_t)}[\mathcal{C}(h_t) \neq y_t]\right| \leq \epsilon.$$

To relate TV distance to KL divergence, we use [Pinsker's inequality](https://en.wikipedia.org/wiki/Pinsker%27s_inequality), which gives $\mbox{TV}(q(h_t \mid x_t),p(h_t \mid x_t)) \leq  \sqrt{\frac{1}{2} KL(q(h_t \mid x_t) \parallel p(h_t \mid x_t))}$. $~~QED.$

This observation explains why solving Task A in practice does not automatically lead to  useful representations for classification tasks (Task C): the posterior distribution was not learnt as accurately as was needed (either due to model mismatch or computational complexity).


## The  link between Tasks A and C: variational methods 

 As noted, distribution learning (Task A) via cross-entropy/maximum-likelihood fitting, and representation learning  (Task C) via sampling the posterior are fairly distinct. Why do students often conflate the two?  Because in practice the most frequent way to solve Task A does  implicitly compute posteriors and thus also solves Task C. 

The generic way to learn latent variable models involves variational methods, which can be viewed as a generalization of the famous EM algorithm  ([Dempster et al. 1977](http://web.mit.edu/6.435/www/Dempster77.pdf)). 

Variational methods maintain at all times a *proposed distribution* $q(h | x)$ (called *variational distribution*). The methods rely on the observation  that for every such $q(h \mid x)$ the following lower bound holds
\begin{equation} \log p(x) \geq E_{q(\mid x)} \log p(x,h) + H(q(h\mid x))  \qquad (2). \end{equation}
where $H$ denotes Shannon entropy (or differential entropy, depending on whether $x$ is discrete or continuous). The RHS above is often called the *ELBO bound* (ELBO = evidence-based lower bound). This inequality follows from a bit of algebra using non-negativity of KL divergence, applied to distributions $q(h \mid x)$ and $p(h\mid x)$. More concretely, the chain of inequalities is as follows, 
$$ KL(q(h\mid x) \parallel p(h \mid x)) \geq 0 \Leftrightarrow E_{q(h|x)} \log \frac{q(h|x)}{p(h|x)} \geq 0 $$
$$ \Leftrightarrow  E_{q(h|x)} \log \frac{q(h|x)}{p(x,h)} + \log p(x) \geq 0 $$ 
$$ \Leftrightarrow \log p(x) \geq  E_{q(h|x)} p(x,h) + H(q(h\mid x)) $$ 
Furthermore, *equality* is achieved if $q(h\mid x) = p(h\mid x)$. (This can be viewed as some kind of "duality" theorem for distributions, and dates all the way back to Gibbs. )

Algorithmically observation (2) is used by foregoing solving the maximum-likelihood optimization (1), and solving instead
$$\max_{\theta, q(h_t|x_t)} \sum_{t} E_{q(h_t\mid x_t)} p(x_t,h_t) + H(q(h_t\mid x_t)) $$ 
Since the variables are naturally divided into two blocks: the model parameters $\theta$, and the variational distributions $q(h_t\mid x_t)$, a natural way to optimize the above is to *alternate* optimizing over each group, while keeping the other fixed. (This meta-algorithm is often called variational EM for obvious reasons.) 

Of course, optimizing over all possible distributions $q$ is an ill-defined problem, so typically one constrains $q$ to lie in some parametric family (e.g., " standard Gaussian transformed by depth $4$ neural nets of certain size and architecture") such that the maximizing the ELBO for $q$ is a tractable problem in practice. Clearly if the parametric family of distributions  is expressive enough, and the (non-convex) optimization problem doesn't get stuck in bad local minima, then variational EM algorithm will give us not only values of the parameters $\theta$ which are close to the ground-truth ones, but also variational distributions $q(h\mid x)$ which accurately track $p(h\mid x)$. But as we saw above, this accuracy would need to be very high to get meaningful representations.

## Next Post

In the next post, we will describe our recent work further clarifying this issue of representation learning via a Bayesian viewpoint.
