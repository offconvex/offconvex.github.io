---
layout: page
title: Off the Convex Path
permalink: /about/
---

##Contributors

* [Sanjeev Arora](http://www.cs.princeton.edu/~arora)
* [Moritz Hardt](http://mrtz.org)
* [Nisheeth Vishnoi](http://theory.epfl.ch/vishnoi/Home.html)

##Contributing an article

If you're writing an article for this blog, please follow these [guidelines](/guide/).

##Submitting an article

The blog lives in this [github repository](https://github.com/offconvex/offconvex.github.io). If you are a regular contributor and your github account has admin access to the repository, you can add an article like this from your command line:

~~~
git clone https://github.com/offconvex/offconvex.github.io.git
cd offconvex.github.io/_posts
cp 2015-12-01-template.md 2015-12-01-your-post-tile.md
git add 2015-12-01-your-post-tile.md
# edit the article using your favorite editor
git commit -m "my post"
git pull
git push
~~~

If you don't have admin access, you can either create a [pull request](https://help.github.com/articles/creating-a-pull-request/) or send any regular contributor the article in markdown. If so, please start from [this template]() and follow the [guidelines](/guide/).
