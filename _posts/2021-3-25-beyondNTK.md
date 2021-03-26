---
layout:     post
title:      When are Neural Networks more powerful than Neural Tangent Kernels?
date:       2021-03-25 14:00:00
author:     Yu Bai, Minshuo Chen, Jason D. Lee
visible:    True
---

The empirical success of deep learning has posed significant challenges to machine learning theory: Why can we efficiently train neural networks with gradient descent despite its highly non-convex optimization landscape? Why do over-parametrized networks generalize well? The recently proposed Neural Tangent Kernel (NTK) theory offers a powerful framework for understanding these, but yet still comes with its limitations.

In this blog post, we explore how to analyze wide neural networks beyond the NTK theory, based on our recent [Beyond Linearization paper](https://arxiv.org/abs/1910.01619) and follow-up [paper on understanding hierarchical learning](https://arxiv.org/abs/2006.13436). (This blog post is also cross-posted at the [Salesforce Research blog](blog.einstein.ai/beyond-ntk/).)

### Neural Tangent Kernels
The Neural Tangent Kernel (NTK) is a recently proposed theoretical framework for establishing provable convergence and generalization guarantees for wide (over-parametrized) neural networks [(Jacot et al. 2018)](https://arxiv.org/abs/1806.07572). Roughly speaking, the NTK theory shows that

* A sufficiently wide neural network trains like a linearized model governed by the derivative of the network with respect to its parameters.
* At the infinite-width limit, this linearized model becomes a kernel predictor with the Neural Tangent Kernel (the NTK).

Consequently, a wide neural network trained with small learning rate converges to 0 training loss and generalize as well as the infinite-width kernel predictor. For a detailed introduction to the NTK, please refer to the earlier [blog post](http://www.offconvex.org/2019/10/03/NTK/) by Wei and Simon.

### Does NTK fully explain the success of neural networks?
Although the NTK yields powerful theoretical results, it turns out that real-world deep learning *do not operate in the NTK regime*: 

* Empirically, infinite-width NTK kernel predictors perform slightly worse (though competitive) than fully trained neural networks on benchmark tasks such as CIFAR-10 [(Arora et al. 2019b)](https://arxiv.org/abs/1904.11955). For finite width networks in practice, this gap is even more profound, as we see in Figure 1: The linearized network is a rather poor approximation of the fully trained network at practical optimization setups such as large initial learning rate [(Bai et al. 2020)](https://arxiv.org/abs/2002.04010). 
* Theoretically, the NTK has poor *sample complexity for learning certain simple functions*. Though the NTK is a universal kernel that can interpolate any finite, non-degenerate training dataset [(Du et al. 2018](https://arxiv.org/abs/1810.02054)[, 2019)](https://arxiv.org/abs/1811.03804), the test error of this kernel predictor scales with the RKHS norm of the ground truth function. For certain non-smooth but simple functions such as a single ReLU, this norm can be exponentially large in the feature dimension [(Yehudai & Shamir 2019)](https://arxiv.org/abs/1904.00687). Consequently, NTK analyses yield poor sample complexity upper bounds for learning such functions, whereas empirically neural nets only require a mild sample size [(Livni et al. 2014)](https://arxiv.org/abs/1410.1141).

<div style="text-align:center;">
<img style="width:700px;" src="http://www.offconvex.org/assets/taylor-plot.png" />
<br>
<i><b>Figure 1.</b>
Linearized model does not closely approximate the training trajectory of neural networks with practical optimization setups, whereas higher order Taylor models offer a substantially better approximation.
</i>
<br>
<br>
</div>

These gaps urge us to ask the following
> **Question**: How can we theoretically study neural networks beyond the NTK regime? Can we prove that neural networks outperform the NTK on certain learning tasks?

The key technical question here is to mathematically understand neural networks operating *outside of the NTK regime*.


## Higher-order Taylor expansion
Our main tool for going beyond the NTK is the *Taylor expansion*. Consider a two-layer neural network with $m$ neurons, where we only train the "bottom" nonlinear layer $W$:

$$
f_{W_0 + W}(x) = \frac{1}{\sqrt{m}} \sum_{r=1}^m a_r \sigma( (w_{0,r} + w_r)^\top x).
$$

(Here, $W_0+W$ is an $m\times d$ weight matrix, where $W_0$ denotes the random initialization and $W$ denotes the trainable "movement" matrix initialized at zero). For small enough $W$, we can perform a Taylor expansion of the network around $W_0$ and get

$$
f_{W_0+W}(x) = \frac{1}{\sqrt{m}} \sum_{r=1}^m a_r \sigma(w_{0,r}^\top x) + \sum_{k=1}^\infty \frac{1}{\sqrt{m}} \sum_{r=1}^m a_r \frac{\sigma^{(k)} (w_{0,r}^\top x)}{k!} (w_r^\top x)^k 
$$

Let us denote the $k$-th order term as $ f^{(k)}_{W_0, W}$, and rewrite this as

$$
f_{W_0+W}(x) = f^{(0)}_{W_0}(x) + \sum_{k=1}^\infty f^{(k)}_{W_0, W}(x).
$$

For the moment also assume that $f^{(0)}_W(x)=0$ (this can be achieved via techniques such as the symmetric initialization).

The key insight of the NTK theory can be described as the following **linearized approximation** property
> For small enough $W$, the neural network $f_{W_0,W}$ is closely approximated by the linear model $f^{(1)}_{W_0,W} = \nabla_W f_{W_0}^\top W$.

Towards moving beyond the linearized approximation, in our [Beyond Linearization paper](https://arxiv.org/abs/1910.01619), we start by asking
> Why just $f^{(1)}$? Can we also utilize the higher-order term in the Taylor series such as $f^{(2)}$?

At first sight, this seems rather unlikely, as in Taylor expansions we always expect the linear term $f^{(1)}$ to dominate the whole expansion and have a larger magnitude than $f^{(2)}$ (and subsequent terms).


### “Killing" the NTK term by randomized coupling
We bring forward the idea of *randomization*, which helps us escape the "domination" of $f^{(1)}$ and couple neural networks with their quadratic Taylor expansion term $f^{(2)}$. This idea appeared first in [Allen-Zhu et al. (2018)](https://arxiv.org/abs/1811.04918) for analyzing three-layer networks, and as we will show also applies to two-layer networks in a perhaps more intuitive fashion. 

Let us now assign each weight movement $w_r$ with a *random sign* $s_r\in\\{\pm 1\\}$, and consider the randomized weights $\\{s_rw_r\\}$. The random signs satisfy the following basic properties:

$$
E[s_r]=0 \quad {\rm and} \quad s_r^2 \equiv 1.
$$

Therefore, let $SW\in\mathbb{R}^{m\times d}$ denote the randomized weight matrix, we can compare the first and second order terms in the Taylor expansion at $SW$:

$$
E_{S} \left[f^{(1)}_{W_0, SW}(x)\right] = E_{S} \left[ \frac{1}{\sqrt{m}}\sum_{r\le m} a_r \sigma'(w_{0,r}^\top x) (s_rw_r^\top x) \right] = 0,
$$

whereas

$$
f^{(2)}_{W_0, SW}(x) = \frac{1}{\sqrt{m}}\sum_{r\le m} a_r \frac{\sigma^{(2)}(w_{0,r}^\top x)}{2} (s_rw_r^\top x)^2 = \frac{1}{\sqrt{m}}\sum_{r\le m} a_r \frac{\sigma^{(2)}(w_{0,r}^\top x)}{2} (w_r^\top x)^2 = f^{(2)}_{W_0, W}(x).
$$

Observe that the sign randomization keeps the quadratic term $f^{(2)}$ unchanged, but "kills" the linear term $f^{(1)}$ in expectation! 

If we train such a randomized network with freshly sampled signs $S$ at each iteration, the linear term $f^{(1)}$ will keep oscillating around zero and does not have any power in fitting the data, whereas the quadratic term is not affected at all and thus becomes the leading force for fitting the data. (The keen reader may notice that this randomization is similar to Dropout, with the key difference being that we randomize the weight *movement* matrix, whereas vanilla Dropout randomizes the weight matrix itself.)

<div style="text-align:center;">
<img style="width:700px;" src="http://www.offconvex.org/assets/beyond-ntk.png" />
<br>
<i><b>Figure 2.</b>
The NTK regime operates in the "NTK ball" where the network is approximately equal to the linear term. The quadratic regime operates in a larger ball where the network is approximately equal to the sum of first two terms, but the linear term dominates and can blow up at large width. Our randomized coupling technique resolves this by introducing the random sign matrix that in expectation "kills" the linear term but always preserves the quadratic term.
</i>
<br>
<br>
</div>

Our first result shows that networks with sign randomization can still be efficiently optimized, despite its now non-convex optimization landscape:
> **Theorem**: Any escaping-saddle algorithm (e.g. noisy SGD) on the regularized loss function $E_S[L(W_0+SW)]+R(W)$, with freshly sampled sign $S=S_t$ per iteration, can find the global minimum in polynomial time.

The proof builds on the quadratic approximation $E_S[f]\approx f^{(2)}$ and recent understandings on neural networks with quadratic activation, e.g. [Soltanolkotabi et al. (2017)](https://arxiv.org/abs/1707.04926) & [Du and Lee (2018)](https://arxiv.org/abs/1803.01206). 

### Generalization and sample complexity: Case study on learning low-rank polynomials
We next study the generalization of these networks in the context of learning *low-rank degree-$p$ polynomials*:

$$
f_\star(x) = \sum_{s=1}^{r_\star} \alpha_s (\beta_s^\top x)^{p_s}, \quad |\alpha_s|\le 1,\|(\beta_s^\top x)^{p_s}\|_{L_2} \le 1, p_s\le p \quad \textrm{for all } s.
$$

We are specifically interested in the case where $r_\star$ is small (e.g. $O(1)$), so that $y$ only depends on the projection of $x$ on a few directions. This for example captures teacher networks with polynomial activation of bounded degree and analytic activation (approximately), as well as constant depth teacher networks with polynomial activations.

For the NTK, the sample complexity of learning polynomials have been studied extensively in [(Arora et al. 2019a)](https://arxiv.org/abs/1901.08584), [(Ghorbani et al. 2019)](https://arxiv.org/abs/1904.12191), and many concurrent work. Combined, they showed that the sample complexity for learning degree-$p$ polynomials is $\Theta(d^p)$, with matching lower and upper bounds:
> **Theorem (NTK)** : Suppose $x$ is uniformly distributed on the sphere, then the NTK requires $O(d^p)$ samples in order to achieve a small test error for learning any degree-$p$ polynomial, and there is a matching lower bound of $\Omega(d^p)$ for any inner-product kernel method.

In our [Beyond Linearization paper](https://arxiv.org/abs/1910.0161), we show that the quadratic Taylor model achieves an improved sample complexity of  $\tilde{O}(d^{p-1})$ with isotropic inputs: 
> **Theorem (Quadratic Model)**: For mildly isotropic input distributions, the two-layer quadratic Taylor model (or two-layer NN with sign randomization) only requires $\tilde{O}({\rm poly}(r_\star, p)d^{p-1})$ samples in order to achieve a small test error for learning a low-rank degree-$p$ polynomial.

In our [follow-up paper on understanding hierarchical learning](https://arxiv.org/abs/2006.13436), we further design a "hierarchical learner" using a specific three-layer network, and show the following
> **Theorem (Three-layer hierarchical model)**: Under mild input distribution assumptions, a three-layer network with a fixed representation layer of width $D=d^{p/2}$ and a trainable quadratic Taylor layer can achieve a small test error using only $\tilde{O}({\rm poly}(r_\star, p)d^{p/2})$ samples.

When $r_\star,p=O(1)$, the quadratic Taylor model can improve over the NTK by a multiplicative factor of $d$, and we can further get a substantially larger improvement of $d^{p/2}$ by using the three-layer hierarchical learner. Here we briefly discuss the proof intuitions, and refer the reader to our papers for more details.

* **Generalization bounds**: We show that, while the NTK and quadratic Taylor model expresses functions using similar random feature constructions, their generalization depends differently on the norm of the input. In the NTK, the generalization depends on the L2 norm of the features (as well as the weights), whereas generalization of the quadratic Taylor model depends on the operator norm of the input matrix features $\frac{1}{n}\sum x_ix_i^\top$ times the nuclear norm of $\sum w_rw_r^\top$. It turns out that this decomposition can match the one given by the NTK (it is never worse), and in addition be better by a factor of $O(\sqrt{d})$ if the input distribution is mildly isotropic so that $\\|\frac{1}{n}\sum x_ix_i^\top\\|_{\rm op} \le 1/\sqrt{d} \cdot \max \\|x_i\\|_2^2$, leading to the $O(d)$ improvement in the sample complexity.

* **Hierarchical learning**: The key intuition behind the hierarchical learner is that we can utilize the $O(d)$ sample complexity gain to its fullest, by applying quadratic Taylor model to not the input $x$, but a feature representation $h(x)\in \mathbb{R}^D$ where $D\gg d$. This yields a gain as long as $h$ is rich enough to express $f_\star$ and also isotropic enough to let the operator norm $\\|\frac{1}{n}\sum h(x_i)h(x_i)^\top\\|_{\rm op}$ be nice. In particular, for learning degree-$p$ polynomials, the best we can do is to choose $D=d^{p/2}$, leading to a sample complexity saving of $\tilde{O}(D)=\tilde{O}(d^{p/2})$.

### Concluding thoughts
In this post, we explored higher-order Taylor expansions (in particular the quadratic expansion) as an approach to deep learning theory beyond the NTK regime. The Taylorization approach has several advantages:

* Non-convex but benign optimization landscape;
* Provable generalization benefits over NTKs;
* Ability of modeling hierarchical learning;
* Convenient API for expeirmentation (cf. the [Neural Tangents](https://github.com/google/neural-tangents) package and the [Taylorized training](https://arxiv.org/abs/2002.04010) paper).

We believe these advantages make the Taylor expansion a powerful tool for deep learning theory, and our results are just a beginning. We also remark that there are other theoretical frameworks such as the [Neural Tangent Hierarchy](https://arxiv.org/abs/1909.08156) or the [Mean-Field Theory](https://arxiv.org/abs/1804.06561) that go beyond the NTK with their own advantages in various angles, but without computational efficiency guarantees. See the [slides](https://jasondlee88.github.io/slides/beyond_ntk.pdf) for more on going beyond NTK. Making progress on any of these directions (or coming up with new ones) would be an exciting direction for future work.
