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


# Principal Components - A Simple Fast Approximate Implementation

I've read more than a few posts about PCA and always end up using the `sklearn` implementation because I don't feel comfortable enough with the topic. Sometimes it's dense formulae, often its technical mathematical papers that would take weeks to digest. The concept of a principal component seems simple enough, then why is a good scalable algorithm for computing them so evasive and complicated? It's actually not.

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

Now the reason we've got the `states` argument to the `forward` function is because we want to study the smallest space that all the data gets projected into (fewer dimensions means less information) throughout the network. If we want to prove convergence, we need to prove that all information is not lost! How do we measure the size of the "space" that the data is occupying? You guessed it, principal component analysis.


## Relationship with the Singular Value Decomposition (SVD)

[//]: # Mathematics is an economy of properties. What I mean by that is we care about defining sets and understanding what logical properties those sets have. To truly appreciate the connection between PCA and SVD, we need to observe that matrices are *linear operators*. To be linear means that
[//]: # - `f(x+y) = f(x) + f(y)`, it is commutative and you can apply any sequence of the operator in any order, and
[//]: # - `f(cx) = c f(x)` for any real number `c`, meaning scalar multiples translate in and out.
[//]: # The fact that a matrix is linear means that it has a maximum and minimum (when its values are bounded)

For a large number of points we often want to know how much "information" they contain. If a lot of three dimension points are actually distributed over a two dimensional disc, then we'd prefer to just store the two dimensional points to save space. When we measure the "information" of a point with the 2-norm, we become interested in the set of directions that maximizes the 2-norm of all data points. And conversely, we become interested in the set of directions in which our data points do not change at all (if any such directions exist). The principal components, or equivalently the right singular vectors, are those directions in which a matrix has the most "information" (or largest magnitude 2-norm).

So then, if this set of directions exists, how do we find it?

The first unintuitive tool that I generally see people use is the `QR` decomposition. People that are very comfortable with linear algebra might find this concept simple, but otherwise it can feel complicated. The concept underneath is quite simple with a visual, consider the following 3 points in three dimensional space:

**TODO: Visual of 3 (not perfectly orthogonal) points.**

What unique directions do these points represent? How can we find them? One way is to start with the largest vector (by 2-norm), and remove the component of all other vectors in that direction. If we follow that process repeatedly removing one vector from all that remains we get this:

**TODO: Visual of iteratively removing components.**

- computers round numbers, so scaling a vector up to be larger will magnify errors (show how that happens)
- instead we want to always proceed from the largest length vector because scaling it less means that we are scaling up our error less
- finally that gives us the following function:

```python
# Given column vectors (in a 2D numpy array), orthogonalize and return
#  the orthonormal vectors and the lengths of the orthogonal components.
def orthogonalize(col_vecs, essentially_zero=2**(-26)):
    rank = 0
    lengths = np.zeros(col_vecs.shape[1])
    for i in range(col_vecs.shape[1]):
        lengths[i:] = np.linalg.norm(col_vecs[:,i:], axis=0)
        # Move the longest vectors to the front / leftmost position (pivot).
        descending_order = np.argsort(-lengths[i:])
        lengths[i:] = lengths[i:][descending_order]
        col_vecs[:,i:] = col_vecs[:,i+descending_order]
        # Break if none of the remaining vectors have positive length.
        if (lengths[i] < essentially_zero): break
        # Normalize then orthogonalize, remove vector i from all remaining vectors.
        col_vecs[:,i] /= lengths[i]
        if (i+1 < col_vecs.shape[1]):
            v = np.dot(col_vecs[:,i],col_vecs[:,i+1:]) * col_vecs[:,i:i+1]
            col_vecs[:,i+1:] -= v
    # Return the orthonormalized vectors and their lengths (before normalization).
    return col_vecs, lengths
```


Once you can orthogonalize a set of vectors, we're actually very close to being able to compute principal components. Most importantly, let's make an observation. Orthogonalizing points should ideally be enough to find out which directions have no variance, but it doesn't tell us which are "most important". However, we know that our matrix will scale vectors that are nearest to the largest principal component the most. In turn, repeatedly applying our matrix and orthogonalizing will make our largest principal component grow faster than all others. With this knowledge, we can implement a simple and fast SVD algorithm.

```python
# Compute the singular values and the right singular vectors for a matrix of row vectors.
def svd(row_vecs, steps=5, bias=1.0):
    dim = row_vecs.shape[1]
    # Initialize a holder for the singular valeus and the right
    #   singular vectors (the principal components).
    # Rescale the data for stability control, exit early when given only zeros.
    multiplier = abs(row_vecs).max()
    assert (multiplier > 0), "ERROR: Provided 'row_vecs' had no nonzero entries."
    multiplier = bias / multiplier
    # Compute the covariance matrix (usually the most expensive operation).
    row_vecs = row_vecs * multiplier
    covariance = np.matmul(row_vecs.T, row_vecs)
    # Compute the initial right singular vectors.
    right_col_vecs, lengths = orthogonalize(covariance.copy())
    # Do the power iteration.
    for i in range(steps):
        right_col_vecs, lengths = orthogonalize(
            np.matmul(covariance, right_col_vecs))
    # Compute the singular values from the lengths.
    singular_vals = lengths
    singular_vals[singular_vals > 0] = np.sqrt(
        singular_vals[singular_vals > 0]) / multiplier
    # Return the singular values and the right singular vectors.
    return right_col_vecs.T, singular_vals
```

To recap, there are only three major pieces that are required to find the principal components:

 1) Compute the (pairwise) covariance of all components of your data.
 2) Orthogonalize the vectors in the covariance matrix (find the zeros!)
 3) Apply the covariance matrix repeatedly to converge on the largest principal components.

In practice, we usually only need approximate principal components. The most important things are the order of the components and the number of zeros. We can actually compute both of these with high reliability in less than a tenth of the time required by the more complicated PCA (and SVD) implementations.

**TODO: Speed comparison with `sklearn` and `numpy.svd`.**
**TODO: Final speed comparison if we implement the above in compiled language (Fortran).**


