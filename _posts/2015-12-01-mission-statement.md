---
layout:     post
title:      Off the convex path
date:       2015-12-01 08:30:00
summary:    What this blog is about
visible:    False
---

Convexity not only underlies a lot of mathematics but also drives modern algorithm design and continuous optimization. While this leads to a beautiful theory, a surprising fact of life is that simple nonconvex optimization often does the job a lot quicker in practice, and better.
Many procedures in statistics and machine learning---reasoning using Bayes nets, or learning deep nets for example---involve solving problems that are nonconvex and in many cases, provably intractable (eg, NP-hard). Yet in practice they can be solved for very large input sizes.  Clearly, there is a mismatch between reality and the predictions of worst-case complexity.
 
A related fact is that many systems in nature---involving neurons, proteins or evolution itself---also seem to compute or employ nonconvex procedures in an efficient manner. Thus understanding what makes these nonconvex methods efficient may help cast light on these natural systems as well.
 
This blog is dedicated to the idea that nonconvex optimization methods---whether created by humans or nature---are exciting objects of study and, often lead to efficient algorithms and deep insights into nature. Of course, this study can be seen as an extension of classical mathematical fields such as dynamical systems and differential equations, but with the important addition of understanding when they are efficient.
 
The blog will report progress that has been made, and highlight interesting research directions and open problems. We will write articles ourselves as well as invite such articles from others.
