---
layout: post
title: Learning a Basis for Approximation
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


# Learning a Basis for Approximation

This post will establish a set of proofs that show useful properties about composed piecewise linear approximations as well as demonstrating that they are capable of approximating a function with a "learned basis", alleviating the problem of choosing a set of basis functions for approximation.

The properties proven here show that the class of composed piecewise linear functions serves as a strong candidate for arbitrary approximation. This is mainly due to the minimum curvature properties that can be achieved with proper initialization and fitting methodologies. Prepared correctly, the resulting fits can converge almost certainly to minimum residual solutions in fixed-dimension approximation scenarios.


## Setting the groundwork

The following considers the problem of function approximation given data `X = {x_1, ..., x_n} ⊂ R^d` and values `{f(x_1), ..., f(x_n)} = {y_1, ..., y_n} = Y ⊂ R^o` for positive integers `d` and `o` that are generated with some true function `f: R^d ➞ R^o` with gradient `𝛿f` that satisfies `||𝛿f(x_i) - 𝛿f(x_j)|| < 𝜀 ||x_i - x_j||` everywhere for some `𝜀 ∈ R` where `||∙||` refers to the 2-norm.

The class of models that will be used to approximate function values given data looks like:

`M(x) = m_s(...m_1(x))` for positive integer `s`, where `m_i(x)` = `max(0, A_i x + b_i)` for real matrix `A_i`, real vector `b_i`, `i ∈ {1,...,s}`, and `x ∈ X`. The shorthand `m_i(X)` will be used to refer to the matrix of values produced by applying `m_i(...m_1(x))` for all `x ∈ X` and `m_{i,j}(X)` will refer to the vector of values associated with the `j`th component of `m_i(X)`.

The error of `M(X)` will be measured in the 2-norm, with the goal to minimize `||M(X) - Y||`. Without loss of generality, we will assume that the components of `Y` have a mean of zero and that `||<y_{1,j}, ..., y_{n,j}>|| = 1`, the components of `Y` have unit 2-norm.


## Theorem

`min_{A_s} ||M(X) - Y||` is invariant to `||m_{s-1,j}(X)||`

### Proof:

Suppose `A_s` is the matrix that produces the minimum norm solution to `A_s m_{s-1}(X) = Y`. Now, since `A_s` is a linear operator, we can commute any scalar multiplier over `m_{s-1,j}(X)` into the columns of `A_s`.

⧠

A similar argument holds for all `||m_{s,j}(X)||`. Knowing this, we can assume `||m_{s,j}(X)|| = 1` without altering the minimum norm solution obtainable by `M(X)`.


## Theorem

The space of basis functions is closed.

### Proof:

This follows directly from the fact that all internal vectors `m_{s,j}(X)` can have norm 1. They all exist on the surface of a ball.

⧠


## Theorem

`||M(X) - Y||` is invariant to rescaled `A_i` 

### Proof:

Since `M` is composed of linear operations `A_i x + b_i`, any multiplicative factors applied to `A_i` can be commuted.

⧠


## Theorem

`||M(X) - m_{i-1,j=0}(X)|| <= ||b_s + A_s(...(b_i + A_i(||m_{i,j}(X)||)))||`

Plainly stated, the 2-norm of the contribution of the `j`th component of `m_i` to the output of the model is upper bounded by the application of all weight matrices and shifts absent truncation.

Since the `b_i` terms will be the same for all other components of `m_i(X)`, relative rankings of the upper bound of contribution of `m_{i,j}(X)` to the 2-norm of output can be computed by simply multiplying the norm of values by all proceeding matrices `A_i`.


## Theorem

The smallest singular value of `m_i(X)` indicates the level of "redundancy" in the basis functions represented at `m_i(X)` with respect to the data. When that smallest singular value equals (or approaches) zero, any one function in `m_{i-1}(X)` can be modified while minimizing change in `M(X)`.


## Theorem

The values captured by a single component `m_{i,j}(X)` can be approximated with some linear combination of `m_{k≠i,j}(X)`, the maximum error associated with the using the approximation can be calculated by projecting the 2-norm of the approximation error through all proceeding transformations.


## Theorem

Larger state spaces approach `Y`

The distance from `Y` to the plane defined by linear projections of `m_s(X)` gets smaller as we add more independent basis vectors to `m_s`.


## Theorem

`||M(X) - Y|| - ||M_{i,j=0}(X) - Y|| < ||m_{i,j}(X)|| 𝜎_{i+1} ... 𝜎_s` for largest singular values `𝜎_i`

This is only true without the unit norm rescaling factors mentioned in a previous theorem. Will need to think more on this.


## Theorem

The magnitude of change in the components of the next layer are a function of the spacing of the data at the current layer. This specific property of data spacing causes increased (or decreased) magnitude change with gradient steps, reducing the curvature of the error function.


## Theorem

Gradient points towards "nearest basis".

The magnitude of the gradient of `||M(X) - Y||` with respect to `m_s(X)` will be largest for the basis function `m_{s,j)(X)` that maximizes `||(Y - M(X)) m_{s,j)(X)||`.

Similarly, the magnitude of the gradient of `||M(X) - Y||` with respect to `m_i(X)` will be largest for the basis function `m_{i,j)(X)` that maximizes `||m_{i+1}^{-1}(...m_s^{-1}(Y - M(X))) m_{i,j)(X)||`.


## Theorem

Upper bounds for curvature

The curvature of loss with respect to a vector in `A_i` is less than or equal to `||m_s(...m_{i+1}(e_i))||`.



## Theorem

The space of parameters is closed.

`||M1 - M2|| < l` such that `||M2(X) - Y|| = 0` and `l ∈ R+`.


## Theorem

Gradient and inverse are connected

The gradient at `m_i(X)` is equal to `m_i(X) - A_{i+1}^{-1}(...A_s^{-1}(Y - b_s)... - b_{i+1})`.  True assuming you can compute "nearest inverse to the input where arbitrary".


## Theorem

Least similar basis is pushed to zero.

Assuming a unit max norm is maintained over vectors in a layer, taking gradient steps will result in the vector that is the least similar with the current error vector being the smallest


## Theorem

Growing solution volume

The volume of the initialization space that converges to a global solution grows with increasing number of basis functions. Global solutions are difficult to find with a small number of truncated linear basis functions.


## Theorem

The curvature of the error function with respect to all pairs of parameters related to the radial symmetry of the data on input. The radial symmetry is connected to the orthogonality of input components. Axis aligned distribution flattening helps when inputs are already orthogonal. Using a separate model to aggregate features that derive from a sampled value function can improve orthogonality of input components. Any "prerequisite information" contained in the positional samples at input can be captured with learnable "location" vectors.
