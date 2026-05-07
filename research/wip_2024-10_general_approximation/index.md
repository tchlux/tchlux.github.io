---
layout: post
title: "Work in progress: The Simplest Approximation for Everything"
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

*Work in progress. This is an incomplete working draft.*

# The Simplest Approximation for Everything

Our reality is predictable, to a large extent. And we are lucky to find ourselves in a place with powerful tools and an ability to make life *better* by leveraging predictions. Breakfast is prepared with fresh ingredients from our refrigerator, flashing lights move vehicular traffic and knowledge through society, and 200 ton rockets are caught after short trips to space. In computer science and statistics, large *language* models are becoming truly helpful assistants for general purpose tasks by doing nothing more than *predicting* the next letters in text. Every one of these currently requires a different and *unique* approximation to be constructed, different methodologies, different models, different perspectives, and different consequences. We will show at a fundamental level they can be unified, and argue for why they should be. So please join me on this thoughtful adventure, and I hope you take away from it some new and powerful ideas of your own.

First we'll try to answer whether there *is* a perspective that unlocks general approximation. We will remain optimistic and assume there are some scenarios where we *should* and that is reason enough to pursue to goal of simplification.

# A diagram of it *all*

- the architecture
- modes of evaluation

numeric and categorical input (known or unknown geometry)
aggregate and fixed input (known or unknown *amounts*)
pairwise comparison as general reasoning

# Converting *Everything* to Geometry

Examples of known geometries.
Examples of unknown things that have simple geometries in finite dimension.
Examples of unknown things that have approximate geometries in finite dimension.

## Approximating the *process*

- initialization
- fitting
- evaluation


# Building on a Reliable Foundation

Linear regression and linear least squares libraries have been reliable tools for scientists for decades. It is possible to achieve similar results for nonlinear models and this work explores one such methodology
for fast, robust, and reliable nonlinear approximation. The idealistic consequence is that relatively "complex" nonlinear models of phenomenon can be approximated with the provided library given little to no data preprocessing or preparation in a as few steps as:

```python3
# Load data from file.
x = loadtxt("input_data.csv")
y = loadtxt("output_data.csv")

# Fit a model that Approximates X -> Y (AXY) given data "x" and values "y".
model = AXY()
model.fit(
    x=x,
    y=y,
)

# Make a prediction at new point(s) "z".
z = loadtxt("test_input.csv")
fy = model.predict(z)
```

# On the Reliable Construction of Nonlinear Approximations

One of the most demonstrably powerful tools in modern applied approximation ("machine learning") is that of parametric models that use a form of gradient guided optimization to fit very large amounts of data. Libraries that make computing the gradient easy for large computational graphs in conjunction with the general proliferation of hardware capable of massively parallel computation have made it possible to tractably and economically fit models that are at *least* Gigabytes ($2^30$ bytes) in size to Terabytes ($2^40$ bytes) or more of data. Such large amounts of data inevitably capture many useful phenomenon, and when the problem is posed correctly such large models do a good job of fitting functions (with unknown underlying structures) to the point that the models themselves become commercially viable products.

In order to fit these large models to large amounts of data, a lot of human engineering effort is required. Almost no model architecture is equally capable in every domain, 

## Problem settings under consideration

- fixed continuous output
- fixed continuous input
- aggregate continuous input
- we will see that almost all approximation problems including classification and textual input are amenable to this setting

------------------------------------------------------------------


# Model and data initialization

For nonlinear approximations it is imperative to prepare the problem well. Particularly, there are a fixed number of nonlinearities in the approximation and in order for nontrivial solutions to be obtained, those nonlinearities need to be uniquely located relative to the data that is being fit.

- we pose the model and the data as large vectors that need to be aligned with each other, minimizing 2-norm
- visual of the data as a vector and the model output as a vector
- visual showing how a "good fit" will have small distance between the vectors
- description that in the limit the problem we are approximating is an infinite vector (a function)


## Well-spacedness and its relationship to model fitting

- visual of a model with nonlinearities located poorly relative to data and the distance between the vectors as well as the minimum distance between vectors
- visual of a model with nonlinearities interspersed throughout the data and the distance between the model and data vector


## Well-spaced initialization for the model

- each layer of the model points in "directions", so these directions should be well spaced
- random normal distribution and its directional uniformity
- orthogonalization doesn't change biases, and prevents unwanted scaling
- linearly spaced nonlinearities provide guaranteed spacing
- proof of the minimum spacing between normalized value vectors when initialized this way
- note that was are assuming some domain, in this case the unit ball


## Compatible and uniform transformations of data

- the model is distributed over the unit ball, so we want the data in the same domain
- when problems are not already scaled nicely, inverting the principal components is the linear transformation that makes the data the "nearest" to being uniformly spaced over the unit ball
- scaling up very small variations in data can be numerically dangerous
- proof that when the random error associated with input data components is relative to the scale of the data, then the inverse principal components makes the error uniform over a ball (not directionally biased) while also making the spacing of the value of the function directionally unbiased (by accounting for scaling differences)
- removing `NaN` and `Inf` values


## Classification as a subset of regression

We treat classification problems as regression problems by regressing the boundary between classes. Without explicitly defining the boundary, but placing each category into a directionally invariant geometry, we show that a form of structurally optimal models will always recover the smoothest boundary between categories.

- one's encoding degeneracy, using the regular simplex mapping
- too many categories by approximate vector representations


## Approximating categorical inputs as continuous

- approximate vector representations
- assume some finite amount of information is contained in categories, which implies it can be approximated with a fixed vector representation


------------------------------------------------------------------

# Model fitting

- generating an approximation is a matter of pointing the model in the same direction as the function
- the error function in the linear scenario is convex
- in the nonlinear scenario, the error function is nonconvex
- as long as the model is entirely nondegenerate, the set of local minima are all "good" approximations


## Truncated second order estimation of the error function

- the error gradient is continuous in value and the magnitude of the gradient is Lipschitz continuous
- exponential trailing mean approximations prevent "bouncing" while remaining a logarithmic number of steps away from a downhill direction
- exponential trailing variance estimations approximate the magnitudes of the columns of the Hessian
- column magnitude estimates for the Hessian function as a diagonal approximation for purposes of minimization


## Adaptive optimization for tail convergence

- shrinking step sizes when error increases guarantees the gradient estimate will converge locally and avoid catastrophic divergence
- increasing step sizes when error decreases maximizes the distance traveled per step, minimizing the number of steps
- using simple heuristics to cut time, steps since best solution versus steps to completion


## Closing the feasible set via conditioning

- if the space of models is unbounded, then we cannot search everywhere
- we only need to find the *direction* of the function, the magnitude is already bounded
- declaring a maximum vector length for each state transformation closes the space of possible models, this is implemented as a max 2-norm on the internal vectors
- similarly the space of approximate vector representations needs to be bounded, and a max 2-norm value is used there, a mean centering is used
- similarly the space of aggregated vector values needs to be bounded, and a max 2-norm value is used there, a mean centering is used
- these could be added in a Lagrangian fashion to the model error, but makes the problem "harder", instead they are treated as independent optimization problems to reduce the size of the full Hessian


## Well spaced validation for accurate error estimation

- error approximations at data points used for fitting are not independent of (orthogonal with) the error function, separate points need to be used for error estimation that were not used for training
- the spacing of data relates closely to the accuracy of an approximation, well spaced data produces a more accurate approximation
- the validation data should be well spaced to maximally minimize error in estimating the error function


## Randomizing and batching in memory constrained environments

- memory usage growth rates relative to model size
- bounding this by selecting subsets of data
- random subset selection, the linear generator
- accelerating convergence by keeping points that "improve" the model


------------------------------------------------------------------

# Allowing aggregate inputs

- some problems have variable size input, so aggregation is permissible


## Fixed input size sets

- fixed input models are a subset of aggregate ones
- categorical component inputs aggregate to relatively correct fixed vectors


## Varying input size sets

- some problems have varying input size, this poses no issues
- mean aggregate vector representations make the approximation invariant to input size, allowing for independence between input size and output


## Pairwise aggregation

- sets are inherently unordered, so order may be introduced either linearly or with categories
- some common problems are invariant to specific combinations, so that is implemented here


------------------------------------------------------------------

# Sample problems and tests

## Fixed inputs, continuous output

## Fixed inputs, categorical output

## Aggregator only model

## Aggregate then approximate

## Mixing continuous and categorical outputs

## Text-based approximation

## Image-based approximation

------------------------------------------------------------------

# Code library

## Standard usage

- `__init__`
- `fit`
- `predict`
- `embed`
- `gradient`


## Internal structure

compiled source (fortran)

- types
- axy
- matrix operations
-- blas (gemm, ...)
-- lapack (gesv, ...)
- normalization
- conditioning
- optimization

convenience wrapper (python)

- Axy class (saving, loading, categorical mappings)
- AxyModel class (for printing and inspection of model)
- Details class (for printing and inspection of fit data and model)


## Compute and Memory profiling

- default timers (for different internal routines)
- using the Details class to inspect sizes


## Development and maintenance trajectory

- unit tests
- Scenarios


## Contributing

- pull requests encouraged
- implementations in other languages greatly desired
- ask before doing major refactors


------------------------------------------------------------------

Cite this post in `BibTeX` with:

```
@incollection{tchlux:axy,
  title     = "The Simplest Approximation for Everything",
  booktitle = "Research Compendium",
  author    = "Lux, Thomas C.H.",
  year      = 2024,
  month     = oct,
  publisher = "GitHub Pages",
  doi       = "10.5281/zenodo.6071692",
  url       = "https://tchlux.info/research/wip_2024-10_general_approximation"
}
```


\bye
