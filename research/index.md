---
layout: post
title: Research Projects `towards the learner`
---

# Research Projects `towards the learner`
[![DOI](https://zenodo.org/badge/221336331.svg)](https://zenodo.org/badge/latestdoi/221336331)

This blog tracks my active research projects. The purpose is twofold: (1) to do good science that is open to the community, and (2) forcing me to formalize, clarify, and polish my thoughts regularly. It's my hope that all work here is complete and correct (unless labeled as in-progress). If you find any problems or errors, please don't hesitate to [raise an issue](https://github.com/tchlux/tchlux.github.io/issues)!


## [Data Distributions and Model Initialization](https://tchlux.github.io/research/2022-03_nn_svd/)

This post accompanies the [interview I did](https://youtu.be/ZaOp1KNhpUQ) with [Machine Learning Street Talk](https://www.youtube.com/c/MachineLearningStreetTalk/). It juxtaposes the consequences of various initialization schemes and grounds their success in geometry. 

## [Aggregate Exchange Models for Approximation](https://tchlux.github.io/research/2022-02_aggregate-exchange-model/)

A take on capturing the approximation power of transformers in a much simpler architecture. It's a **work-in-progress**, will keep adding as things are refined.

## [Fast Principal Component and Rank Approximation](https://tchlux.github.io/research/2022-02_principal_components/)

**Work-in-progress:** This casual explainer walks through a simple, fast, approximate implementation of principal component analysis in Python + NumPy that is faster than `sklearn` PCA and `numpy.linalg` SVD. The implementation only assumes an ability to do matrix multiplication.


## [Controlling bias with data from other problems](https://tchlux.github.io/research/2022-02_smoothing-with-artificial-data/)

An alternative to pretraining that guarantees a model captures some desirable underlying structure for problems where very little specific data is available, but data for a similar domain is available in abundance.


## [MLP's are actually nonlinear ➞ linear preconditioners](https://tchlux.github.io/research/2021-10_mlp_nonlinear_linear_preconditioner/)

Sometimes it's easy to overlook the fundamental nature of neural networks with all their complexity. Almost all architectures reduce to performing linear regression in the last layer, and that's more useful than it may seem.

# Citing this Work

Refer to the entire research blog in `BibTeX` with
```
@book{tchlux:research,
  title     = "Research Compendium",
  author    = "Lux, Thomas C.H.",
  year      = 2022,
  publisher = "GitHub Pages",
  doi       = "10.5281/zenodo.6071692",
  url       = "https://tchlux.info/research"
}
```

Or refer to any specific post by substituting `<post-title>` and `<post-url>` appropriately within
```
@incollection{tchlux:research,
  title     = "<post-title>",
  booktitle = "Research Compendium",
  author    = "Lux, Thomas C.H.",
  year      = 2022,
  publisher = "GitHub Pages",
  doi       = "10.5281/zenodo.6071692",
  url       = "https://tchlux.info/research/<post-url>"
}
```


[//]: # ## [Mechanical Groundwork](https://tchlux.github.io/research/2022-02_getting_started/)
[//]: # Testing out the mechanics of posting to this site. How do things look? Does a multiline summary beneath the posts make sense right here? Only one way to find out.
