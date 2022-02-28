---
layout: post
title: Principal Components the Easy Way
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


# Principal Components - A Simple Fast Implementation

I've read more than a few posts about PCA and always end up using the `sklearn` implementation because I don't feel comfortable enough with the topic. Sometimes it's dense formulae, often its technical mathematical papers that would take weeks to digest. The concept of a principal component seems simple enough, then why is a good scalable algorithm for computing them so evasive and complicated? It's not.

Let's finally write our own PCA algorithm and build the intuition that comes with coding every step.

### Background

I'm currently exploring ways to make multilayer perceptrons (MLP's) provably converge. It's always bothered me that initialization seems so arbitrary and all the optimization algorithms produce different results that are "better for `X` domain" or "optimal for `N` or bigger data". That's a whole other post in the making.

Let's consider a simple `MLP` that looks like:

```python
import numpy as np

# Configure the model.
input_dim = 2
output_dim = 1
state_dim = 32
num_states = 8

# Initialize the weights (input, internal, output).
weight_matrices = [np.random.normal(size=(input_dim, state_dim))] + \
                  [np.random.normal(size=(state_dim, state_dim))
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

Now the reason we've got the `states` argument to the `forward` function is because we want to study the smallest space that all the data gets projected into (fewer dimensions means less information) throughout the network. If we want to prove convergence we need to prove that all information is not lost! How do we measure the size of the "space" that the data is being represented in? You guessed it, principal component analysis.


