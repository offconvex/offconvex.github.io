---
layout:     post
title:      Semantic Word Embeddings
date:       2015-12-01 9:00:00
summary:    Understanding word embeddings
author:     Sanjeev Arora
visible:    False
---

This post can be seen as an introduction to how nonconvex problems arise
naturally in practice, and also the relative ease with which they are often
solved. 

I will talk about *word embeddings*, a geometric way to capture
the “meaning” of a word via a low-dimensional vector. They are useful in
many tasks in Information Retrieval (IR) and Natural Language Processing
(NLP), such as answering search queries or translating from one
language to another. 

You may wonder: how can a 300-dimensional vector capture the many
nuances of word meaning? And what the heck does it mean to "capture meaning?"

##Properties of Word Embeddings
A simple property of embeddings obtained by all the methods I’ll
describe is *cosine similarity*: the  *similarity* between two words 
(as rated by humans on a $[-1,1]$ scale) correlates with the *cosine*
of the angle between their vectors. To 
give an example, the cosine for *milk* and
*cow* may be $0.6$, whereas for *milk* and
*stone* it may be $0.2$, which is roughly the similarity
human subjects assign to them.


A more interesting property of recent embeddings is that they can solve
*analogy* relationships via linear algebra.
For example, the word analogy question
*man : woman ::king : ??* can be solved by looking for the
word $w$ such that $v_{king} - v_w$ is most similar to
$v_{man} - v_{woman}$; in other words, minimizes


$$||v_w - v_{king} + v_{man} - v_{woman}||^2.$$

This simple idea can solve $75\%$ of analogy questions on some standard testbed.
Here is a rendering of this linear algebraic relationship between *masculine-feminine* pairs.

![linear](/assets/analogy-small.jpg)

Good embeddings have other properties that will be covered in a future
post. (Also, I can't resist mentioning  that word embeddings seem related to how the human brain
encodes meaning; see the [well-known paper of Mitchell et al.](http://www.cs.cmu.edu/~tom/pubs/science2008.pdf).) 

##Computing Word embeddings (via Firth's Hypothesis)

In all the methods, the word vector is a succinct representation of the distribution of other words around this word. That this suffices to capture meaning is asserted by [*Firth’s hypothesis*](https://en.wikipedia.org/wiki/Distributional_semantics)
from 1957, "*You shall know a word by the company it keeps.*"  To give an example, if I
ask you to think of a word that tends to co-occur with *cow,
drink, babies, calcium*, you would immediately answer:
*milk*. 

Note that we don't believe Firth’s hypothesis fully accounts
for all aspects of semantics. (If it it did, a computer would be able to
completely learn language  in an unsupervised way by processing a large
enough text corpus. Experts ---even those who believe 
computers will ultimately pass the [Turing test](https://en.wikipedia.org/wiki/Turing_test)---doubt such unsupervised learning suffices: for example, understanding new metaphors or jokes
seems to require some physical experience of the real world.) But Firth's hypothesis does imply a very simple 
word embedding, albeit a very high-dimensional one.


> *Embedding 1*: Suppose the dictionary has $N$ distinct words (in practice, $N =100,000$). Take a very large text corpus (e.g., Wikipedia) and let $Count_5(w_1, w_2)$ be the number of times $w_1$ and $w_2$ occur within a distance $5$ of each other in the corpus. Then the word embedding for a word $w$ is a vector of dimension $N$, with one coordinate for each dictionary word. The coordinate corresponding to word $w_2$ is $Count_5(w, w_2)$.


The obvious problem with Embedding 1 is that it uses
extremely high-dimensional vectors. How can we compress them?

> *Embedding 2*: Do dimension reduction by taking the rank-300
> singular value decomposition (SVD) of the above vectors. 

Recall that for an $N \times N$ matrix $M$ this means finding vectors
$v_1, v_2, \ldots, v_N \in \mathbb{R}^{300}$ that minimize

$$
\sum_{ij} (M_{ij} - v_i \cdot v_j)^2 \qquad (1).
$$

Using SVD to do dimension reduction seems an obvious idea these days but it
actually is not. After all,  it is unclear *a priori* why the
above $N \times N$ matrix of cooccurance counts should be close to a
rank-300 matrix. That this is the case was empirically discovered in the paper on
[*Latent Semantic Analysis* or LSA](http://lsa.colorado.edu/papers/plato/plato.annote.html), which in fact shows an even stronger empirical fact: dimension reduction via SVD *improves* the quality of
the embeddings. (Improvement via compression is a familiar phenomenon in machine learning.) 

A research area called *Vector Space Models* (see [survey by
Turney and Pantel](https://www.jair.org/media/2934/live-2934-4846-jair.pdf)) studies various modifications of the above idea, most
of which involve reweighting the above raw counts: some buzzwords are
TF-IDF, PMI, Logarithm, Square-root, etc.. In general reweighting the
$(i, j)$ term in expression (1) leads to a *weighted*
version of SVD, which is NP-hard. (I always emphasize to my students
that a polynomial-time algorithm to compute rank-k  SVD is a miracle, since 
modifying the problem
statement in small ways makes it NP-hard.) But in practice, weighted SVD can be
solved ---like many other nonconvex problems in machine learning---on a laptop in less than a day
by simple gradient descent on the objective (1), possibly also using
a regularizer. (Of course,  a lot has been proven about such methods in context of [convex optimization](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html); the surprise is that they work in this nonconvex setting.)  Remember, the matrix is $N \times  N$, i.e., rather
large!

But to the best of my knowledge, the following question had not been
raised or debated: *What property of human language explains the
fact that these very high-dimensional matrices of weighted word counts
are approximable by low-rank matrices?* (In a subsequent blog
post I will describe our new [theoretical explanation](http://arxiv.org/abs/1502.03520).)

The third embedding method I wish to describe uses 
*energy-based models*, for instance the
[**Word2Vec** family of methods](https://code.google.com/p/word2vec/) from 2013 by the Google team of Mikolov et al., which also created a buzz due to the above-mentioned linear algebraic method to solve word analogy tasks.
The **word2vec** models are inspired by pre-existing neural net models for language
(basically, the word embedding corresponds to the neural net's internal representation of the word; see [this blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/).).  Let me
describe the simplest variant, which assumes that the word
vectors are related to word probabilities as follows:


> *Embedding 3* (**Word2Vec(CBOW)**):
>
> $$
> \Pr[w|w_1, w_2, \ldots, w_5] \propto \exp(v_w \cdot (\frac{1}{5} \sum_i v_{w_i})),\qquad (2)
> $$
>
> where the left hand side gives the empirical probability that word $w$ occurs in the text
conditional on the last five words being $w_1$ through $w_5$. 

Assume we can estimate the left hand side using a large text corpus (which actually we can't since the corpus is not large enough; **word2vec** uses a short cut called *negative sampling*, which I won't describe). Then expression (2) for the word vectors---together with  a constraint capping the dimension of the
vectors to, say, 300 --- implicitly defines a nonconvex optimization problem. Such models arise in  host of domains and  seem solvable
at fairly large scales. 


The **word2vec** papers are a bit mysterious, and have motivated much
followup work. A paper by Levy and Goldberg (See [Omer Levy's Blog](https://levyomer.wordpress.com/)) explains that the **word2vec**
methods are actually modern versions of older vector space methods.
After all, if you take logs of both sides of expression (2), you see
that the *logarithm* of some cooccurence probability is
being expressed in terms of inner products of some word vectors, which
is very much in the spirit of the older work. (Levy and Goldberg have more to say, and also interesting experiments.) 

Another paper by Pennington et al. at Stanford suggests a [model called GLOVE](http://nlp.stanford.edu/projects/glove/)
that uses an explicit weighted-SVD strategy for finding word embeddings.
They also give an intuitive explanation of why these embeddings solve 
word analogy tasks, though the explanation isn't quite rigorous. 

In a future post I will talk more about our [subsequent work](http://arxiv.org/abs/1502.03520) that tries to unify these different approaches, and also explains some  cool properties of word
embeddings.
