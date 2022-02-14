---
layout: post
title: MLP's are actually nonlinear ➞ linear preconditioners
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

# MLP's are actually nonlinear ➞ linear preconditioners

In spirit of yesterday being a [bones day](https://www.tiktok.com/@jongraz/video/7022251358833118469), I put together a few visuals last night to show off something people might not always think about. Enjoy!

Let's pretend our goal was to approximate this function with data.

<p class="visual">
  <video controls="" autoplay="" loop="" type="video/mp4" src="https://preview.redd.it/9nwp1rueofv71.gif?format=mp4&s=54da61c3d6ce2b15fa56a77d5c23ffdf336c93db"></video>  
</p>
<p class="caption"><code>cos(norm(x))</code> over <code>[-4π, 4π]</code></p> 


To demonstrate how a neural network "makes a nonlinear function linear", here I trained a 32 × 8 [multilayer perceptron](https://github.com/tchlux/tchlux.github.io/blob/dfe4c113826bbefca41109b9b0c8697b3e00e9e7/documents/piecewise_linear_regression_model.f90) with [PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html) activation on the function `cos(norm(x))` with a random uniform 10k points over the `[-4π, 4π]` square. The training was done with 1k steps of full-batch Adam (roughly, [my own version of Adam](https://github.com/tchlux/tchlux.github.io/blob/master/documents/piecewise_linear_regression_model.f90#L664-L696)). Here's the final approximation.

<p class="visual">
  <video controls="" autoplay="" loop="" type="video/mp4" src="https://preview.redd.it/ji8ykw1iofv71.gif?format=mp4&s=9545be2288363378c852918d58459f2188ea1b36"></video>
</p>
<p class="caption">(32 × 8) PReLU MLP approximation to <code>cos(norm(x))</code> with 10k points</p> 


Not perfect, but pretty good! Now here's where things get interesting. What happens if you look at the "last embedding" of the network, what does the function look like in that space? Here's a visual where I've taken the representations of the data at that last layer and projected them onto the first two [principal components](https://setosa.io/ev/principal-component-analysis/) with the true function value as the z-axis.

<p class="visual">
  <video controls="" autoplay="" loop="" type="video/mp4" src="https://preview.redd.it/0zt6443kofv71.gif?format=mp4&s=bfd7cd955b6e8fa86adb40eae19988fe35fcfd04"></video>
</p>
<p class="caption">Last-layer embedding of the 10k training points for the MLP approximating <code>cos(norm(x))</code></p> 


Almost perfectly linear! To people that think about what a neural network does a lot, this might be obvious. But I feel like there's a new perspective here that people can benefit from:

## When we train a neural network, we are constructing a function that nonlinearly transforms data into a space where the curvature of the "target" is minimized!

In numerical analysis, transformations that you make to data to improve the accuracy of later approximations are called "preconditioners". Now preconditioning data for linear approximations has many benefits other than just minimizing the loss of your neural network. [Proven error bounds](https://tchlux.github.io/documents/tchlux-2020-thesis-slides-theorem.pdf) for piecewise linear approximations (many neural networks) are affected heavily by the curvature of the function being approximated (full proof is in [Section 5 of this paper](https://tchlux.github.io/papers/tchlux-2020-NUMA.pdf) for those interested).


### *What does this mean though?*

It means that after we train a neural network for *any* problem (computer vision, natural language, generic data science, ...) we don't have to use the last layer of the neural network (*ahem*, linear regression) to make predictions. We can use k-nearest neighbor, or a [Shepard interpolant](https://en.wikipedia.org/wiki/Inverse_distance_weighting), and the accuracy of those methods will usually be improved significantly! Check out what happens for this example when we use k-nearest neighbor to make an approximation.

<p class="visual">
  <video controls="" autoplay="" loop="" type="video/mp4" src="https://preview.redd.it/bz4ssu2nofv71.gif?format=mp4&s=4a4338ef5ab93e36bb5e3370cc1691f9853a8b21"></video>
</p>
<p class="caption">Nearest neighbor approximation to <code>3x+cos(8x)/2+sin(5y)</code> over unit cube.</p> 


Now, train a small neural network (8×4 in size) on the \~40 data points seen in the visual, transform the entire space to the last layer embedding of that network (8 dimensions), and visualize the resulting approximation back in our original input space. This is what the new nearest neighbor approximation looks like.

<p class="visual">
  <video controls="" autoplay="" loop="" type="video/mp4" src="https://preview.redd.it/xg5rageoofv71.gif?format=mp4&s=11a7fa0bfbfdc9e7b71b23c29a6849f21c7379ad"></video>
</p>
<p class="caption">Nearest neighbor over the same data as before, but after transforming the space with a small trained neural network.</p> 


[Pretty neat!](https://youtu.be/Hm3JodBR-vs) The maximum error of this nearest neighbor approximation decreased significantly when we used a neural network as a preconditioner. And we can use this concept *anywhere*. Want to make distributional predictions and give statistical bounds for any data science problem? Well that's really easy to do with lots of nearest neighbors! And we have all the tools to do it.


> ***About me:*** *I spend a lot of time thinking about how we can progress towards useful digital intelligence (AI). I do not research this full time (maybe one day!), but rather do this as a hobby. My current line of work is on building theory for solving arbitrary approximation problems, specifically investigating a generalization of transformers (with nonlinear attention mechanisms) and how to improve the convergence / error reduction properties & guarantees of neural networks in general.*  
>  
>*Since this is a hobby, I don't spend lots of time looking for other people doing the same work. I just do this as fun project. Please share any research that is related or that you think would be useful or interesting!*
