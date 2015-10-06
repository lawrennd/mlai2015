---
layout: page
title: "Overview"
---

COM4509/COM6509 Machine Learning and Adaptive Intelligence 2015-16
==================================================================


### Course Overview

This unit aims to provide an understanding of the fundamental technologies underlying modern artificial intelligence. In particular it will provide foundational understanding of probability and statistical modelling, supervised learning for classification and regression, and unsupervised learning for data exploration. The teaching consists of two hours of lectures and one of lab classes each week. The lectures are on Tuesdays, the labs on Fridays. The teaching schedule and venue for each week are given below:

# Lectures

{% for post in site.posts reversed %}
- {{ post.title }} {% if post.time %} {{ post.time }} {% endif %} on {{ post.date | date: "%b %-d, %Y" }}{% if post.venue %} in **{{ post.venue }}**{% endif %}.{% if post.tagline %} [*{{ post.tagline }}*]({{ post.url | prepend: site.baseurl }}).{% endif %}
{% endfor %}

Past Papers
-----------

Information on past papers is [available](./coursePastPapers.html).

<p class="rss-subscribe">subscribe <a href="{{ "/feed.xml" | prepend: site.baseurl }}">via RSS</a></p>
