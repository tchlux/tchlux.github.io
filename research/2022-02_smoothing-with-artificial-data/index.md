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


# Controlling bias with data from other problems
<p class="caption">Find the source code <a href="https://github.com/tchlux/tchlux.github.io/tree/master/research/2022-02_smoothing-with-artificial-data/code.py">here</a>, uses <code>tlux=0.0.4</code>. </p>

I've recently been working on supporting "learned embeddings" for categorical inputs to my [Fortran MLP](https://github.com/tchlux/util/blob/master/util/approximate/plrm/stable/stable_relu.f90). When running some tests I noticed a very interesting result!

This is a function I usually use as a sanity-check whenever I implement a new approximation algorithm.

<p class="visual">
 <iframe src="./test-function.html">
 </iframe>
</p>
<p class="caption"><code>3x + cos(8x)/2 + sin(5y)</code> over the unit cube <code>[0,1], [0,1]</code>.</p>


This is what a typical fit of a tine sample (`N = 15`) might look like for a small MLP architecture (ReLU activation, 16 nodes per layer, 8 layers):

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
<p class="caption">16×8 ReLU MLP fit of 15 points (dots) from <code>3x + cos(8x)/2 + sin(5y)</code> and 300 points from a piecewise linear interpolant of the first 15 points over the unit cube using 1000 steps to fit with approximate 2nd order minimization. Scroll down to see model error throughout fit.</p>


Now this fit may or may not look better in this example, but the important take away is that we can use some additional categorical input to bias the output of the model for our target problem. And we can use dense data from a similar domain even when we have very little training data!

This suggests that for large vision or natural language problems where we may not have much data at all (few-shot learning), we can include data and predicted outputs from other similar problems to bias the model towards "meaningful representations".

The most common alternative to this that I am aware of is pretraining a model with more dense data from a different problem that has similar structure. Both of these processes work, but the internal model structures created in pretraining can be undone with later training that doesn't include the pretraining data. Alternatively, this categorical encoding approach ensures that the internal structures are maintained through training. If convergence suffers, then weights can be placed on the target task to ensure the quality of the fit on the target task is not significantly degraded.
