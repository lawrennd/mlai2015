---
layout: post
title: Week 1
date: 2015-09-29 09:00
venue: SB LT-2
tagline: Introduction to machine learning and a review of probability theory
type: lecture
labclass: week1.ipynb
lecturepdf: w1_probability.pdf
lecturenb: w1_probability.ipynb
---

Data Science Overview and Jupyter Introduction
==============================================

Lecture Notes
-------------

{{ page.tagline }} \[{%if page.lecturepdf %}[PDF Lecture slides]({{ site.url }}/assets/{{ page.lecturepdf }}){% endif %}\]\[{%if page.lecturenb %}[Jupyter Lecture slides]({{ site.nbviewer }}/{{ page.lecturenb }}){% endif %}\] 

Lab Class
---------

The notebook for the lab class can be downloaded from
[here]({{ site.nbviewer }}/{{ page.labclass }}).

To obtain the lab class in ipython notebook, first open the ipython
notebook. Then paste the following code into the ipython notebook

    import urllib.request
    urllib.request.urlretrieve('{{ site.gitraw }}/{{ page.labclass }}', '{{ page.labclass }}')

You should now be able to find the lab class by clicking `File->Open` on
the jupyter notebook menu.

