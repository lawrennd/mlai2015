---
layout: page
title: "Overview"
---

COM4509/COM6509 Machine Learning and Adaptive Intelligence 2015-16
==================================================================


### Course Overview

This unit aims to provide an understanding of the fundamental technologies underlying modern artificial intelligence. In particular it will provide foundational understanding of probability and statistical modelling, supervised learning for classification and regression, and unsupervised learning for data exploration. The teaching consists of two hours of lectures and one of lab classes each week. The lectures are on Tuesdays, the labs on Fridays. The teaching schedule and venue for each week are given below:

# Tutorials

{% for post in site.posts reversed %}
- {{ post.title }} {% if post.time %} {{ post.time }} {% endif %} on {{ post.date | date: "%b %-d, %Y" }}{% if post.venue %} in **{{ post.venue }}**{% endif %}.{% if post.tagline %} [*{{ post.tagline }}*]({{ post.url | prepend: site.baseurl }}).{% endif %}
{% endfor %}

1.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week1.html)
2.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week2.html)
3.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week3.html)
4.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week4.html)
5.  [Reading Week](./week5.html)
6.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week6.html)
7.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week7.html)
8.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week8.html)
9.  [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week9.html)
10. [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week10.html)
11. [Reading Week](./week11.html)
12. [Tuesday 9-10 SB-LT2; Tuesday 11-13 MAPP-F110 Lab Class](./week12.html)


Lecture Slides: Draft Schedule
------------------------------

The material for the lectures will be updated before each lecture (including audio and screen capture, where possible). As a place holder last year's information is currently stored.

- [Week 1: Uncertainty and Probability](./assets/w1_uncertaintyAndProbability.pdf)
  Introduction to Pandas, Jupyter and Probability

- [Week 2: Objective Functions and Recommender Systems](./assets/w2_objective.pdf)
  Recommender systems Lab

- [Week 3: Linear Algebra and Regression](./assets/w3_regression.pdf)
  Linear Algebra and Regression in Python

- [Week 4: Basis Functions](./assets/w4_basisFunctions.pdf)
  Basis Functions Lab

- Week 5: Reading Week

- [Week 6: Generalization](./assets/w6_generalisation.pdf)
  Generalization Lab

- [Week 7: Bayesian Regression](./assets/w7_bayesianRegression.pdf)
  Bayesian Regression Lab

- [Week 8: Dimensionality Reduction](./assets/w8_dimensionalityReduction.pdf)
  Dimensionality Reduction Lab

- [Week 9: Classification: Naive Bayes](./assets/w9_classification.pdf)
  Classification Lab

- Week 10: Classification: Logistic Regression
  Classification Lab

- Week 11: Reading Week and Question and Answer

- [Week 12: Gaussian Processes](./assets/w12_gaussianProcesses.pdf)
  Gaussian Process Lab

Past Papers
-----------

Information on past papers is [available](./coursePastPapers.html).

