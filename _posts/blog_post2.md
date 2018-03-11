---
layout: post
title: Limitations of Encoder-Decoder GAN architectures
date:  2018-2-24 16:00:00
author: Sanjeev Arora and Andrej Risteski
visible: False
---
This is yet another new post about [Generative Adversarial Nets (GANs)](http://www.offconvex.org/2017/03/15/GANs/), and based upon our new ICLR'18 paper.  A quick recap of the story so far. GANs are an unsupervised method in deep learning to learn interesting distributions (e.g., images of human faces), and also have a plethora of uses for image-to-image mappings in computer vision. Standard GANs training is motivated using this task of distribution learning, and is designed with the idea that given large enough deep nets and enough training examples, as well as accurate optimization, GANs will learn the full distribution. 

 [Sanjeev's previous post](http://www.offconvex.org/2017/03/30/GANs2/) concerned [his co-authored ICML'17 paper](https://arxiv.org/abs/1703.00573) which called this intuition into question when the deep nets have finite capacity. It shows that the training objective has near-equilibria where the discriminator is fooled ---i.e., training objective is good---but the generator's distributions has very small support, i.e. shows *mode collapse.*  This is a failure of the model, and raises the question whether such bad equilibria are found in real-life training. A [second post](http://www.offconvex.org/2017/07/07/GANs3/) showed empirical evidence that they do, using the birthday-paradox test. 

The current post concerns our [new result](https://arxiv.org/abs/1711.02651) (part of our upcoming [ICLR paper](https://openreview.net/forum?id=BJehNfW0-)) which shows that bad equilibria exist also in more recent GAN architectures based on simultaneously learning an *encoder* and *decoder*. This should be surprising because many researchers believe that encoder-decoder architectures fix many issues with GANs, including mode collapse.

As we will see, encoder-decoder GANs indeed seem very powerful. In particular, the proof of the previously mentioned [negative result](http://www.offconvex.org/2017/03/30/GANs2/) does not apply to this architecture. But, we then discovered a cute argument that shows encoder-decoder GANs can have poor solutions, featuring not only mode collapse but also encoders that map images to nonsense (more precisely Gaussian noise). This is the worst possible failure of the model one could imagine.

##Encoder-decoder architectures

Encoders and decoders have long been around in machine learning in various forms -- especially deep learning. Speaking loosely, underlying all of them are two basic assumptions:   
(1) Some form of the so-called [*manifold assumption*](https://mitpress.mit.edu/sites/default/files/titles/content/9780262033589_sch_0001.pdf) which asserts that high-dimensional data such as real-life images lie (roughly) on a low-dimensional manifold. ("Manifold" should be interpreted rather informally -- sometimes this intuition applies only very approximately sometimes it's meant in a "distributional" sense, etc.)    
(2) The low-dimensional structure is "meaningful": if we think of an image $x$ as a high-dimensional vector and its "code" $z$ as its coordinates on the low-dimensional manifold, the code $z$ is thought of as a "high-level" descriptor of the image.   

With the above two points in mind, an *encoder* maps the image to its code, and a *decoder* computes the reverse map. (We also discussed encoders and decoders in [our earlier post on representation learning](http://www.offconvex.org/2017/06/27/unsupervised1/) in a more general setup.)  


<p style="text-align:center;">
<img src="/assets/BIGAN_manifold2.jpg" width="80%"  alt="Manifold structure" />
</p>

Encoder-decoder GANs were introduced by [Dumoulin et al.(ALI)](https://arxiv.org/abs/1606.00704) and [Donahue et al.(BiGAN)](https://arxiv.org/abs/1605.09782). They involve two competitors: Player 1 involves a discriminator net $D$ that is given an input of the form (image, code) and it outputs a number in the interval $[0,1]$, which denotes its "satisfaction level" with this input. Player 2 trains a decoder net $G$ (also called *generator* in the GANs setting) and an encoder net $E$.  
<p style="text-align:center;">
<img src="/assets/BIGAN_2player.jpg" width="80%"  alt="Encoder-Decoder Gans, the two players" />
</p>

Player 1 is trying to train its net to distinguish between the following two settings, and Player 2 is trying to make sure the two settings look indistinguishable to Player 1's net. 

$$ \mbox{Setting 1: presented with}~(x, E(x))~\mbox{where $x$ is random real image}.$$
$$ \mbox{Setting 2: presented with}~(G(z), z)~\mbox{where $z$ is random code}.$$

(Here it is assumed that a random code is a vector with i.i.d gaussian coordinates, though one could consider other distributions.)

<p style="text-align:center;">
<img src="/assets/BIGAN_2settings.jpg" width="100%" alt="Two settings which discriminator net has to distinguish between" />
</p>
Notice the difference over vanilla GANs, in which the discriminator merely tries to distinguish real images from images generated by the generator $G$. The hoped-for equilibrium obviously is one where generator and encoder are inverses of each other: $E(G(z)) \approx z$ and $G(E(x)) \approx x$, and the joint distributions $(z,G(z))$ and $(E(x), x)$ roughly match.
The intuition is that if this happens, Player 1 must've produced a "meaningful" representation $E(x)$ for the images -- and this should improve the quality of the generator as well. 
Indeed, [Dumoulin et al.(ALI)](https://arxiv.org/abs/1606.00704) provide some small-scale empirical examples on mixtures of Gaussians for which encoder-decoder architectures seem to ameliorate the problem of mode collapse. 



Both of the above papers prove that when the encoder/decoder have infinite capacity, the equilibrium is indeed the desired one. However, in our paper, we show that the finite capacity case is very different. (Paralleling the state of affairs in the vanilla GAN setup.) 

## Finite-capacity discriminators are weak

As indicated above, we prove that there is a Player 1 (so a generator/encoder pair) that fools all Players 2 that have bounded capacity (i.e. number of parameters), yet they are poor in the appropriate sense: namely, the encoder just outputs white noise (i.e. does not extract meaningful features) and the generator outputs a uniformly random image from a small pool of images (i.e. is not diverse). 

To state the result a bit formally, we say a generator/encoder pair $(G,E)$ $\epsilon$-fools a decoder $D$ if 

$$|E_{x} D(x, E(x)) - E_{z} D(G(z), z)| \leq \epsilon$$
  
(In other words, the output of $D$ in Setting 1 and Setting 2 is nearly the same.)

We show: 

>**Theorem**: There exists a Player 1 $(G,E)$ that $\epsilon$-fools all $L$-Lipschitz discriminators $D$ with at most $p$ parameters, such that:      
>**(1)** The generator $G$ outputs a uniformly random image from a pool of $O(p \log^2(pL)/ \epsilon^2)$ images. (In particular, it is theoretically no more diverse than it would be in the vanilla GAN setting.)    
>**(2)** The encoder $E$ merely "extracts noise" from the images. Thus, it does not produce any "meaningful" representation of the image.        

Let's describe what the generator/encoder in the above theorem look like. 

Let's consider the encoder first: one of the reasons that encoder-decoder GANs seem very powerful is that the discriminator ought to be able to learn to simulate $E$ if it needs to, and thus acquire the ability to apply $E$ to the output of $G$.  This suggests that training should enforce $E$ to be a much smaller deep net than the discriminator. This was one conceptual hurdle for us: how to exhibit bad solutions where $E$ is a smaller net than the discriminator?  

We ensure this with a simple trick: we will assume the image distribution is mildly "noised": say, every 100th pixel is replaced by Gaussian noise. To a human, such an image would of course be indistinguishable from a real image. (The proof could be carried out via some other assumptions to the effect that images have an innate stochastic/noise component that is efficiently extractable by a small neural network. But let's keep things clean.) When noise $\eta$ is thus added to an image $x$, we denote the resulting image as $x \odot \eta$. 

The theorem above uses the trivial encoder $E$ that, given the noised image $x \odot \eta$, outputs $\eta$. Clearly, such an encoder does not in any sense capture "meaning" in the code. It is also implementable by a tiny single-layer net, as needed.

The generator $G(z)$ will have a "pool" of $m := p \log^2(pL)/ \epsilon^2$ unnoised images $\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_m$, and will partition the noise space (for $z$) into $m$ equal-measured blocks. Then, when presented with an input $z$, the generator will output the image $\tilde{x}_i \odot z$, where $i$ is the block $z$ belongs to. (See the Figure below.) 

<p style="text-align:center;">
<img src="/assets/BIGAN_construction.jpg" width="70%" alt="The bad generator construction" />
</p>

## Few words about the proof 

Since we already explicitly specified the encoder $E$, our job is to only prove a good generator $G$ exists. 

The idea is to construct a distribution over generators $G$ that works "in expectation", and use concentration bounds that this implies there must be at least one generator that does the job. To motivate the distribution, notice that the distribution of the pair $(x, E(x))$ is very easy to describe: it's simply $(\tilde{x} \odot z,z)$, where $\tilde{x}$ and $z$ are independent samples from the unnoised image distribution and noise distribution respectively. 

Thus, a natural choice for the distribution on $G$ would be to take the pool of samples 
$\tilde{x}_1, \tilde{x}_2, .., \tilde{x}_m$ for $G$ to be uniformly randomly chosen from the (unnoised) image distribution. 
Why is this distribution for $G$ good? Notice the following simple fact: 

$$E_{G} E_{z} D(G(z), z) =  E_{\tilde{x}, z} D(\tilde{x} \odot z, z) = E_{x} D(x, E(x)) \hspace{2cm} (3)$$ 

In other words, the "expected" encoder correctly matches the expectation of $D(x, E(x))$, so that the discriminator is fooled.
This of course is not enough: we need some kind of concentration argument to show a particular $G$ works, which will ultimately use the fact that the discriminator $D$ has a small capacity. 

 
Towards that, another useful observation: if $q$ is the uniform distribution over sets $T= \{z_1, z_2,\dots, z_m\}$, s.t. each $z_i$ is independently sampled from the conditional distribution inside the $i$-th block of the partition of the noise space, by the law of total expectation one can see that 
$$ E_{z} D(G(z), z) = E_{T \sim q} \frac{1}{m} \sum_{i=1}^m D(G(z_i), z_i) $$
The right hand side is an average of terms, each of which is a bounded function of mutually independent random variables -- so, by e.g. McDiarmid's inequality it concentrates around it's expectation, which by (3) is exactly $E_{z} D(G(z), z)$.

To finish the argument off, we use the fact that due to Lipschitzness and the bound on the number of parameters, the "effective" number of distinct discriminators is small, so we can union bound over them. (Formally, this translates to an epsilon-net + union bound argument. This also gives rise to the value of $m$ used in the construction.)

## Some conclusions 

While the number of new GAN architectures grows by the day, the issue of diversity/mode collapse seems to be quite difficult to overcome -- both theoretically and in practice. Of course, the main questions about GANs still remain: can they be engineered to be truly distribution learners, and if not, what are they best suited for? 

 
 
