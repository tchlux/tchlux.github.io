---
layout: post
title: Smoothing an MLP with extra data.
---

[//]: # (Formula can be generated at:
[//]: #   https://latex.codecogs.com/svg.image?latex_math_mode_code
[//]: # 
[//]: # Images can be included like this:
[//]: #   <img class="formula" src="./local-file.svg" title="name"/>
[//]: # 
[//]: # Visuals in the local director can be included like this:
[//]: #   <p class="visual">
[//]: #   <iframe src="./local-file.html">
[//]: #   </iframe>
[//]: #   </p>
[//]: #   <p class="caption">Caption under the visual.</p>
[//]: # 
[//]: # Everything else follows normal markdown syntax.


# Controlling MLP bias with linear interpolants
<p class="caption">Find the source code <a href="https://github.com/tchlux/tchlux.github.io/tree/master/blog/2022-02_smoothing-with-artificial-data/code.py">here</a>, uses <code>tlux=0.0.4</code>. </p>

I've recently been working on supporting "learned embeddings" for categorical inputs to my [Fortran MLP](https://github.com/tchlux/util/blob/master/util/approximate/plrm/stable/stable_relu.f90). When running some tests I noticed a very interesting result!

This is a function I usually use as a sanity-check whenever I implement a new approximation algorithm.

<p class="visual">
 <iframe src="./test-function.html">
 </iframe>
</p>
<p class="caption"><code>3x + cos(8x)/2 + sin(5y)</code> over the unit cube <code>[0,1], [0,1]</code>.</p>

This is what a typical fit of a tine sample (`N = 15`) might look like for a small MLP architecture (15 nodes per layer, 8 layers):

<p class="visual">
 <iframe src="./fit-surface.html">
 </iframe>
</p>
<p class="caption">16×8 MLP fit of 15 points (dots) from <code>3x + cos(8x)/2 + sin(5y)</code> over the unit cube using 1000 steps to fit with approximate 2nd order minimization. Scroll down to see model error throughout fit.</p>


When I was testing the ability to add different outputs based on a categorical input, I used a dense sample of predictions from a piecewise linear interpolant (Delaunay) over the same 15 points as a "second category". So the 15 original points were put in category 1 while category 2 contained 300 points from the piecewise linear fit. This totaled 315 "training" points for the model over two different categories.

<p class="visual">
 <iframe src="./fit-surface-delaunay.html">
 </iframe>
</p>
<p class="caption">16×8 MLP fit of 15 points (dots) from <code>3x + cos(8x)/2 + sin(5y)</code> and 300 points from a piecewise linear interpolant of the first 15 points over the unit cube using 1000 steps to fit with approximate 2nd order minimization. Scroll down to see model error throughout fit.</p>


