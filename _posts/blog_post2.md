---
layout: post
title: Limitations of Encoder-Decoder GAN architectures
date:  2018-2-24 16:00:00
author: Sanjeev Arora and Andrej Risteski
visible: False
---

This is another post about [Generative Adversarial Nets (GANs)](http://www.offconvex.org/2017/03/15/GANs/).  In [a previous post](http://www.offconvex.org/2017/03/30/GANs2/) Sanjeev talked about the result from [his coauthored ICML'17 paper](https://arxiv.org/abs/1703.00573)  that when generator and discriminator have finite capacity, the training
objective always has bad (near) equilibria where the discriminator is fooled but the generator's distributions has low support, i.e. shows *mode collapse.* This raised the question whether such bad equilibria occur in real-life training, and the last post showed empirical evidence that they do, quantified using the [birthday-paradox test](http://www.offconvex.org/2017/07/07/GANs3/). 

The current post concerns our [new result](https://arxiv.org/abs/1711.02651) (part of our upcoming [ICLR paper](https://openreview.net/forum?id=BJehNfW0-)) which shows that  bad equilibria exist also in more advanced GANs based on simultaneously-trained *encoder* and *decoder.* 

Like most people, when we first saw encoder-decoder GANs we  figured that the discoverers had cracked the mode-collapse puzzle. 
The proof of the [above negative result](http://www.offconvex.org/2017/03/30/GANs2/)  completely breaks for this architecture and fixing the proof seemed hopeless at first.

But we then discovered a cute argument that shows encoder-decoder GANs have bad solutions, featuring not  only mode collapse but also encoders that map images to nonsense (i.e., gaussian noise). This is the worst possible failure of the model one could imagine.

##Encoder-decoder architectures

Encoders and decoders have long been around in machine learning --especially deep learning. Underlying them is the so-called [*manifold assumption*](https://mitpress.mit.edu/sites/default/files/titles/content/9780262033589_sch_0001.pdf) which asserts that high-dimensional data such as real-life images lie (roughly) on a low-dimensional
manifold. (Here "manifold" refers to the [usual math object](https://en.wikipedia.org/wiki/Manifold), though note that ML applications  involve no nontrivial topology or geometry.) Imagine the image $x$ as a high-dimensional vector and its "code" $z$ represents its coordinates on the  manifold, which is a low-dimensional surface. Code $z$ can be thought of as a "high-level" descriptor of the image. An *encoder* maps the image to its code, and a *decoder* computes the reverse map. (We also discussed encoder and decoder  in [our earlier post on representation learning](http://www.offconvex.org/2017/06/27/unsupervised1/), where they were allowed to be one-to-many mappings.)  


<p style="text-align:center;">
<img src="/assets/GANmanifold.jpg" width="80%"  alt="Manifold Assumption and Encode-Decoder GAN" />
</p>

Encoder-decoder GANs were introduced by [Dumoulin et al.(ALI)](https://arxiv.org/abs/1606.00704) and [Donahue et al.(BiGAN)](https://arxiv.org/abs/1605.09782). They involve two competitors: Player 1 involves a discriminator net that is given an input of the form (image, code) and it outputs a number in the interval $[0,1]$, which denotes its satisfaction level with this input.
(In other words, does it find $z$ a reasonable code for this image $x$?)  Player 2 trains a decoder net $G$ (also called *generator* in the GANs setting) and an encoder net $E$.  Player 1 is trying to train its net to distinguish between the following two settings, and Player 2 is trying to make sure the two settings look indistinguishable to Player 1's net. 

$$ \mbox{Setting 1: presented with}~(x, E(x))~\mbox{where $x$ is random real image}.$$
$$ \mbox{Setting 2: presented with}~(G(z), z)~\mbox{where $z$ is random code}.$$

(Here it is assumed that a random code is a vector with iid gaussian coordinates, though one could consider other distributions.)


<p style="text-align:center;">
<img src="/assets/GANbigan.jpg" width="100%" alt="The Bigan setup" />
</p>


The hoped-for equilibrium obviously is one where generator and encoder are inverses of each other ($E(G(z)) \approx z$ and $G(E(x)) \approx x$) and the joint distributions $(z,G(z))$ and $(E(x), x)$ roughly match. This indeed happens when all nets in the above picture have infinite capacity, as proved in  both the above papers.  But now we see that the finite capacity case is very different.


## Finite-capacity discriminators are weak

The intuition behind the working of encoder-decoder GANs is that the discriminator ought to be able to learn to simulate $E$ if it needs to, and thus acquire the ability to apply $E$ to the output of $G$.  In particular, this suggests that training should enforce $E$ to be a much smaller deep net than the discriminator. This was the main conceptual hurdle for us: how to exhibit bad solutions where $E$ is a smaller net than the discriminator?  Ultimately a simple trick ensured this.

First,  a minor assumption. We will assume the image distribution is mildly "noised": say, every 100th pixel is replaced by Gaussian noise. To a human, such an image would of course be pretty indistinguishable from a real image. (The proof could be carried out via some other assumptions to the effect that images have an innate stochastic/noise component that is efficiently extractable by a small deep net. But let's keep things clean.) When noise $\eta$ is thus added to an image $x$, we denote the resulting image, with some abuse of notation, as $x +\eta$. 

Our equilibrium involves the trivial encoder $E$ that, given the noised image $x+\eta$, outputs $\eta$. Clearly, such an encoder dashes all hopes of ever finding any "meaning" in the code. It is also implementable by a tiny single-layer deep net, as needed.

The proof will show that there is a generator $G$ such that to a finite-capacity discriminator, the distribution of pairs $G(z), z$ where $z$ is a random gaussian vector is essentially indistiguishable from the distribution of pairs $x +\eta, \eta$. In other words, the generator wins against the game. 



But, this property can be badly violated when the discriminator has bounded capacity: namely there exists a generator $G$ which outputs a uniformly random image from a pool of $O(p \log(pL)/ \epsilon^2)$ images and yet, it $\epsilon$-fools all $L$-Lipschitz discriminators with outputs in $[0,1]$ and at most $p$ parameters (= number of weights), where $\epsilon$-fooling is defined as 

$$|E_{x} D(x) - E_{z} D(G(z))| \leq \epsilon \hspace{2cm} (1) $$

In the case of encoder-decoder architectures, the infinite capacity case, as mentioned above, was handled by both  [Dumoulin et al. (ALI)](https://arxiv.org/abs/1606.00704) and [Donahue et al. (BiGAN)](https://arxiv.org/abs/1605.09782). However, we show that when the capacity is finite, 

>**Theorem**: There exists a generative player $(G,E)$ that $\epsilon$-fool all $L$-Lipschitz discriminators $D$ with outputs in $[0,1]$ and at most $p$ parameters, such that:      
>**(1)** The generator $G$ outputs a uniformly random image from a pool of $O(p \log^2(pL)/ \epsilon^2)$ images. Thus, it is essentially no more diverse than what we would get in the usual GAN setting.       
>**(2)** The encoder $E$ merely "extracts noise" from the images. Thus, it does not produce any "meaningful" representation of the image.       

Note the definition of $\epsilon$-fooling is a straightforward generalization of (1), namely: 
$$|E_{x} D(x, E(x)) - E_{z} D(G(z), z)| \leq \epsilon \hspace{2cm} (2)$$

This theorem shows that the situation for encoder-decoder architectures mirrors the one in standard GAN architectures. Moreover, the [birthday-paradox](http://www.offconvex.org/2017/07/07/GANs3/) empirical test supports this theoretical prediction, in the sense that BiGAN's and ALI seem to not lead to a substantially greater diversity. 

## The generator/encoder construction

Let's give a bit of an idea of how the pair $(G,E)$ above is constructed. 

The generator $G(z)$ will have a "pool" of $m$ unnoised images $\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_m$, and will partition the noise space (for $z$) into $m$ equal-measured blocks. Then, when presented with an input $z$, the generator will output the image $\tilde{x}_i \odot z$, where $i$ is the block $z$ belongs to, and $\odot$ is the noising operator, which swaps every 100th pixel of $x$ by noise which is in turn denoted as a vector $z$. (See the Figure below.) 

<p style="text-align:center;">
<img src="/assets/bigan/blog_construction.jpg" width="70%" alt="The generator construction" />
</p>
 
The encoder $E$ will be very simple: since the input image distribution is noised, it will output the pixels of the image which correspond to the noised coordinates. Notice that this automatically takes care of point (2) of the main theorem: the "representation" is pure white noise, completely unrelated to the image. Also notice that we are not cheating by making the encoder unreasonably powerful in comparison to the discriminator (i.e. the reason the discriminator is fooled is not computational power differences): in fact, it's easy to see that an encoder of the type above can be represented by a very simple ReLU network. (See our paper for details.)

## Brief sketch of the proof 

We briefly sketch the proof of the main theorem. The main idea is to construct a distribution over generators $G$ that works "in expectation", and use concentration bounds that this implies there must be at least one generator that does the job. 
Towards that, recall the condition for a generator $G$ to $\epsilon$-fool a discriminator $D$ is

$$|E_{x} D(x, E(x)) - E_{z} D(G(z), z)| \leq \epsilon$$

Let us consider the expectations $E_x(x,E(x))$ and $E_z(G(z),z)$ one by one. 

Given the definition of the encoder $E$, the distribution of the pair $(x, E(x))$ is very easy to describe: it's simply $(\tilde{x} \odot z,z)$, where $\tilde{x}$ and $z$ are independent 
samples from the unnoised image distribution and noise distribution respectively. 

To analyze the distribution $(G(z), z)$, we will use the fact generator $G$ is randomly chosen. More concretely, suppose the pool of samples 
$\tilde{x}_1, \tilde{x}_2, .., \tilde{x}_m$ for $G$ is sampled uniformly at random from the (unnoised) image distribution. This of course, defines a distribution over generators of the kind we described above. Notice the following simple fact: 

$$E_{G} E_{z} (G(z), z) =  E_{\tilde{x}, z} D(\tilde{x} \odot z, z) = E_{x} D(x, E(x)) \hspace{2cm} (3)$$ 

In other words, the "expected" encoder correctly matches the expectation of $D(x, E(x))$, so that the discriminator is fooled.
This of course is not enough: we need some kind of concentration argument to show a particular $G$ works, which will ultimately use the fact that the discriminator $D$ has a small capacity.
 
Towards that, let's define $m:= p \log^2(pL)/ \epsilon^2$, and let $T_{\mbox{nc}}$ be the uniform distribution over sets $T= \{z_1, z_2,\dots, z_m\}$, s.t. each $z_i$ is independently sampled from the conditional distribution inside the $i$-th block of the partition of the noise space. By a bit of term rearranging and the law of total expectation, one can see that 
$$ E_{z} D(G(z), z) = E_{T \sim T_{\mbox{nc}}} \frac{1}{m} \sum_{i=1}^m D(G(z_i), z_i) $$
This may perhaps seem like trivial term rearranging, but in fact, the later expression is easier to handle now: $\frac{1}{m} \sum_{i=1}^m D(G(z_i), z_i)$ is an average of terms, each of which has the form $f_i(z_i, \tilde{x}_i)$ for some bounded function $f$, and the random variables $x_i$, $z_i$ are all mutually independent! 

Since $D(\cdot)$ is a function bounded in $[0,1]$, we can use e.g. McDiarmid's inequality to argue about the concentration of $\frac{1}{m} \sum_{i=1}^m D(G(z_i), z_i)$ around it's expectation, which by (3) is exactly $E_{z} D(G(z), z)$. To finish the argument off, we use the fact that due to Lipschitzness and the bound on the number of parameters, the "effective" number of distinct discriminators is small, so we can union bound over them. (Formally, this translates to an epsilon-net + union bound. This also gives rise to the value of $m$ used in the construction.)

## Some conclusions 

While the number of new GAN architectures grows by the day, the issue of diversity/mode collapse seems to be quite difficult to overcome -- both theoretically and in practice. Of course, the main questions about GANs still remain: can they be engineered to be truly distribution learners, and if not, what are they best suited for? 

 
 