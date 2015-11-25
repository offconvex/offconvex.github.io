---
layout:     post
title:      Semantic Word Embeddings
date:       2015-12-01 9:00:00
summary:    Understanding word embeddings
categories: language words embeddings meaning
---

This post can be seen as an introduction to why nonconvex problems arise
naturally in practice, and also the relative ease by which they are often
solved. 

 The topic is *word embeddings*, a geometric way to capture
the “meaning” of a word via a low-dimensional vector. They are useful in
many tasks in Information Retrieval (IR) and Natural Language Processing
(NLP), for example answering search queries or translating from one
language to another. 


You may wonder: how can a 300-dimensional vector capture the many
nuances of word meaning? And what the heck does this mean?


A simple property of embeddings obtained by all the methods I’ll
describe is *cosine similarity*: the  *similarity* between two words 
(as rated by humans on a $$[-1,1]$$ scale) correlates with the *cosine*
of the angle between their vectors. To 
give an example, the cosine for *milk* and
*cow* may be $$0.7$$, whereas for *milk* and
*stone* it may be $$-0.1$$, which is roughly the similarity
human subjects assign to them.


A more interesting property in recent embeddings is that they can solve
*analogy* relationships via linear algebra on embeddings.
For example, the word analogy question
*man : woman ::king : ??* can be solved by looking for the
word $$w$$ such that $$v_{king} - v_w$$ is most similar to
$$v_{man} - v_{woman}$$; in other words, minimizes


$$||v_w - v_{king} + v_{man} - v_{woman}||^2.$$ 

This simple idea can solve $$75\%$$ of analogy questions on some standard testbed.


![linear](/assets/linearrelations.jpg)


Good embeddings have other properties that will be covered in a future
post. Now let’s discuss simple methods to construct them.

The methods all ultimately rely on *Firth’s hypothesis*
from 1954, which says that Namely, the meaning of a word is determined
by the distribution of other words around it. To give an example, if I
ask you to think of a word that tends to co-occur with *cow,
drink, babies, calcium*, you would immediately answer:
*milk*.

Note that I am not suggesting that Firth’s hypothesis fully accounts
for all aspects of semantics. (If it it did, a computer would be able to
learn language completely in an unsupervised way by processing a large
enough text corpus. Many experts doubt this is possible: they feel
language –e.g. understanding new metaphors or jokes— requires some
knowledge of the physical world, as well as some $2$-way interaction
between teacher and student.) But it does imply a very simple 
word embedding, albeit a very high-dimensional one.


> *Embedding 1*: Suppose the dictionary has $$N$$ distinct words (in practice, $$N =100,000$$). Take a very large text corpus (e.g., Wikipedia) and let $$Count_5(w_1, w_2)$$ be the number of times $$w_1$$ and $$w_2$$ occur within a distance $$5$$ of each other in the corpus. Then the word embedding for a word $$w$$ is a vector of dimension $$N$$, with one coordinate for each dictionary word. The coordinate corresponding to word $$w_2$$ is $$Count_5(w, w_2)$$.


The obvious problem with Embedding 1 is that it uses
extremely high-dimensional vectors. How can we compress them?

> *Embedding 2*: Do dimension reduction by taking the rank-$$300$$
> singular value decomposition (SVD) of the above vectors. 

Recall that for an $$N \times N$$ matrix $$M$$ this means finding vectors
$$v_1, v_2, \ldots, v_N \in \mathbb{R}^{300}$$ that minimize

$$
\sum_{ij} (M_{ij} - v_i \cdot v_j)^2 \qquad (1).
$$

Using SVD to do dimension reduction seems an obvious idea today but it
actually is not since it is unclear *a priori* why the
above $$N \times N$$ matrix of cooccurance counts should be close to a
rank-300 matrix. This was empirically discovered in the paper on
*Latent Semantic Analysis* or LSA. (It shows in fact that
low-dimensional embeddings are *better* than
high-dimensional ones.)

A research area called *Vector Space Models* (see survey by
Turney and Pantel) studies various modifications of the above idea, most
of which involve reweighting the above raw counts: some buzzwords are
TF-IDF, PMI, Logarithm, Square-root, etc. In general reweighting the
$$(i, j)$$ term in expression (1) leads to a *weighted*
version of SVD, which is NP-hard. (I always emphasize to my students
what a miraculous algorithm SVD is, since modifying the problem
statement in small ways makes it NP-hard.) But in practice, it can be
solved by simple gradient descent on the objective in (1) —possibly with
a regularizer—which is the usual method to solve nonconvex problems in
machine learning. Remember, the matrix is $$N \times  N$$, i.e., rather
large!

But to the best of my knowledge, the following question had not been
raised or debated: <span>*What property of human language explains the
fact that these very high-dimensional matrices of weighted word counts
are approximable by low-rank matrices?*</span> (In a subsequent blog
post I will describe our new theoretical explanation.)

The third embedding method I wish to desribe uses so-called
*energy-based models*, exemplified by the famous papers on
**Word2Vec** around 2013 by the Google team of Mikolov et al..
They were inspired by pre-existing language models based upon neural nets. Let me
describe the simplest of the many variants. It assumes that the word
vectors are related to word probabilities as follows:


> *Embedding 3* (**Word2Vec(CBOW)**):
>
> $$
> \Pr[w|w_1, w_2, \ldots, w_5] \propto \exp(v_w \cdot (\frac{1}{5} \sum_i v_{w_i})),\qquad (2)
> $$
>
> where the left hand side gives the empirical probability that word $$w$$ occurs in the text
conditional on the last five words being $$w_1$$ through $$w_5$$. 

(Note that there is an implicit constraint capping the dimension of the
vectors at, say, 300.) Such models may look bizarre when you first see
them, but they work well in a host of applications. They are fitted
using nonconvex optimization, on very large corpora. (The optimization
involves a clever idea called <span>*negative sampling*</span>, but I
won’t go into details.)

In fact, the application of embeddings to word analogy tasks is also due
to the **word2vec** papers, and the reason why this word
created a big buzz.

The **word2vec** papers are a bit mysterious, and have motivated much
followup work. A paper by Levy and Goldberg explains that the **word2vec**
methods are actually modern versions of older vector space methods.
After all, if you take logs of both sides of expression (2), you see
that the *logarithm* of some cooccurence probability is
being expressed in terms of inner products of some word vectors, which
is very much in the spirit of the older work.

A paper by Pennington et al. at Stanford suggests a model called GLOVE
that uses an explicit weighted-SVD strategy for finding word embeddings.
In a future post I will talk more about the subsequent work; why word
embeddings solve analogy tasks; and some other cool properties of word
embeddings.

