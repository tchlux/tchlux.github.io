---
layout: post
title: Data Distributions and Initialization
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


# Data Distributions and Initializing Neural Networks

<p class="caption">
  <a href="https://www.reddit.com/">Comment and discuss this post on Reddit!</a>
  <br><a href="https://github.com/tchlux/tchlux.github.io/blob/master/research/2022-03_nn_svd/index.md?plain=1">Raise any issues on GitHub.</a>
  <br><a href="https://github.com/tchlux/tchlux.github.io/blob/master/research/2022-03_nn_svd/run.py">Run this code for yourself.</a>
</p>


I'm currently exploring ways to make multilayer perceptrons (MLP's) provably converge. It's always bothered me that initialization seems so arbitrary and all the optimization algorithms produce different results that are "better for `X` domain" or "optimal for `N` or bigger data". The first step to really studying this phenomenon is to actually look at some data and see what is happening.

Let's consider a simple `MLP` that looks like:

```python3
# Configure the model.
input_dim = 10
output_dim = 1
state_dim = 40
num_states = 10

# Initialize the weights (input, internal, output).
weight_matrices = [normal(input_dim, state_dim)] + \
                  [normal(state_dim, state_dim)
                   for i in range(num_states-1)] + \
                  [np.zeros((state_dim, output_dim))]
shift_vectors = [np.linspace(-1, 1, state_dim) for i in range(num_states)]

# Define the "forward" function for the model that makes predictions.
# Optionally provided "states" matrix that has a shape:
#   (num_states, x.shape[0], state_dim)
def forward(x, states=None):
    for i in range(num_states):
        x = np.clip(np.matmul(x, weight_matrices[i]) + shift_vectors[i],
            0.0, float("inf"))
        if (states is not None):
            states[i,:,:] = x[:,:]
    return np.matmul(x, weight_matrices[-1])
```

Given this simple ReLU architecture, lets define some random uniform
data and construct a test function for us to approximate. We'll use
this to observe some properties of the model.

```python3
# Define some random data.
x = ball(100, input_dim)
y = np.cos(np.linalg.norm(x, axis=1, keepdims=True))

# Initialize holder for the states.
states = np.zeros((num_states, x.shape[0], state_dim))
# Initial evaluation of model at all data points.
forward(x, states=states)
```

Now we've got the internal representations of the data at every layer
of the network. From here we can look at their distributions to see
where we are losing (or gaining) information. Here's a visual of the
sample points that we've generated:

<p class="visual">
 <iframe src="./input_data.html">
 </iframe>
</p>
<p class="caption">Data distribution before it is passed through the model.</p>

What about if we look at the representations that are created for data
inside the model? Below are visuals for layers 1, 5 (middle), and 10 (last.

<p class="visual">
 <iframe src="./normal_init/data_layer_1.html">
 </iframe>
</p>
<p class="caption">Data distribution at the first layer of the model.</p>

<p class="visual">
 <iframe src="./normal_init/data_layer_6.html">
 </iframe>
</p>
<p class="caption">Data distribution at the sixth (middle most) layer of the model.</p>

<p class="visual">
 <iframe src="./normal_init/data_layer_10.html">
 </iframe>
</p>
<p class="caption">Data distribution at the tenth (last) layer of the model.</p>

The first thing to notice is that the scale of the data becomes huge!
This is exactly the problem that schemes like [Kaiming
initialization](https://arxiv.org/abs/1502.01852) try to solve. Random
normal weight initializations create vectors that rescale your
data. This is actually the distribution of 2-norm magnitudes for
weights that have been initialized with values from a purely random
normal distribution:

<p class="visual">
 <iframe src="./normal_init/lengths.html">
 </iframe>
</p>
<p class="caption">2-norm distribution of random normal weight vectors.</p>


A simple way to try and solve that problem is to initialize
weight vectors inside the network to have unit 2-norm (this stops them
from scaling the data up or down, only doing directional projections).
This is what happens when we try to do that.

## Using random <code>sphere</code> initializations

<p class="visual">
 <iframe src="./sphere_init/data_layer_1.html">
 </iframe>
</p>
<p class="caption">Data distribution at the first layer of the model.</p>

<p class="visual">
 <iframe src="./sphere_init/data_layer_6.html">
 </iframe>
</p>
<p class="caption">Data distribution at the sixth (middle most) layer of the model.</p>

<p class="visual">
 <iframe src="./sphere_init/data_layer_10.html">
 </iframe>
</p>
<p class="caption">Data distribution at the tenth (last) layer of the model.</p>


The scaling problem is mostly gone! If anything we need to worry about
the data getting scaled down, but over many repeated trials it seems
to hold more closely to the original scale.

Now the remaining issue is how distorted the data has become. Notice
that our data that was nicely distributed over the unit ball has not
been pushed into a flat arc-like pattern (we've lost a lot of input
direction variance). With this low dimensional input data, the
distortions are not much of an issue, but when we have data that has
high intrinsic dimension (≥ 10 nonzero principal components), then we
can quickly lose important information!


## Chasing the distributions

Let's look at the distribution of magnitudes of the singular values
(the amount of variance of the data along the principal components)
at each of the internal state representations for the model when we
raise the input dimension to 10.

