---
layout: page
title: Contributing an article
permalink: /guide/
---

##Using Markdown

Here's the basic template you're article should use:

~~~
---
layout:     post
title:      Title of your post
date:       2015-12-01 9:00:00
summary:    One line summary
categories: topic1 topic2 topic3
author:     Your name
---

Articles are written in markdown (kramdown). 
Please use minimal formatting for your article.

This is a single paragraph. It is separated by newlines. 
No need for html tags. A single line break does not start a new
paragraph.

This is a second paragraph. 
You can place subsections as follows.

## Using maths

You can use math as you would in latex. Let $n>2$ be an integer.
Consider the equation:
\[
a^n + b^n = c^n
\]

Wiles proved the following theorem.

> **Theorem** (Fermat's Last Theorem).
> The above equation has no solution in the positive integers.

## Links and images

You can place a link like [this](http://wikipedia.org).

You place a picture similarly:

![Lander](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg/320px-Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg)

## Emphasis and boldface

Use *this* for emphasis and **this** for boldface.

## You're all set.

There really isn't more you need to know.

~~~

The body of the article would be rendered as:

***

Articles are written in markdown (kramdown). 
Please use minimal formatting for your article.

This is a single paragraph. It is separated by newlines. 
No need for html tags. A single line break does not start a new
paragraph.

This is a second paragraph. 
You can place subsections as follows.

## Using maths

You can use math as you would in latex. Let $n>2$ be an integer.
Consider the equation:
\[
a^n + b^n = c^n
\]

Wiles proved the following theorem.

> **Theorem** (Fermat's Last Theorem).
> The above equation has no solution in the positive integers.

## Links and images

You can place a link like [this](http://wikipedia.org).

You place a picture similarly:

![Lander](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg/320px-Albert_Bierstadt_-_The_Rocky_Mountains%2C_Lander%27s_Peak.jpg)

## Emphasis and boldface

Use *this* for emphasis and **this** for boldface.

## HTML tags

You can use them if you get stuck, but please avoid them as much as you can. 
Markdown rendering will stop between html tags.

## You're all set.

There really isn't more you need to know.

***
