---
layout: page
title: Week 3
tagline: Linear algebra and regression
venue: SB-LT2
date: 2015-10-13 09:00
time: "9:00"
type: lecture
labclass: week3.ipynb
lecturepdf: w3_regression.pdf
youtube:
---

{% if page.youtube %}
<iframe width="{{ site.youtube.width }}" height="{{ site.youtube.height }}" src="https://www.youtube.com/embed/{{ page.youtube }}" frameborder="0" allowfullscreen></iframe>
{% endif %}

{{ page.tagline }} \[{%if page.lecturepdf %}[PDF Lecture slides]({{ site.url }}{{ site.baseurl }}/assets/{{ page.lecturepdf }}){% endif %}\]\[{%if page.lecturenb %}[Jupyter Lecture slides]({{ site.nbviewer }}/{{ page.lecturenb }}){% endif %}\] 

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

### YouTube Video

There is a YouTube video available of me giving this material at the
[Gaussian Process Road Show in Uganda](http://gpss.cc/gprs13/). You
will need to watch this in HD to make the maths clearer.

Lab Class
---------

Linear regression with numpy and Python.

The notebook for the lab class can be downloaded from
[here](http://nbviewer.ipython.org/github/lawrennd/mlai2015/blob/master/week3.ipynb).

To obtain the lab class in ipython notebook, first open the ipython
notebook. Then paste the following code into the ipython notebook

    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/lawrennd/mlai2015/master/week3.ipynb', 'week3.ipynb')

You should now be able to find the lab class by clicking `File->Open` on
the ipython notebook menu.

### Reading

-   Reading (Regression)
    -   Sections 1.1-1.3 of Rogers and Girolami.
    -   Section 1.2.5 of Bishop up to Eq 1.65.
    -   Section 1.1 of Bishop.
-   Reading (Linear Algebra, Matrix and Vector Review)
    -   Section 1.3 of Rogers and Girolami.
    -   [Linear Algebra Guide](http://betterexplained.com/articles/linear-algebra-guide/)
-   Reading (Basis Functions)
    -   Chapter 1, pg 1-6 of Bishop.
    -   Section 1.4 of Rogers and Girolami.
    -   Chapter 3, Section 3.1 of Bishop up to pg 143.

