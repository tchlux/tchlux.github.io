import time
import numpy as np


# Given a plot object and a series name, add a bunch of lines for "row_vecs".
def plot_vecs(p, name, row_vecs, **kwargs):
    # Break into components.
    components = [
        sum(([0,v,None] for v in row_vecs[:,i]),[])
        for i in range(row_vecs.shape[1])
    ]
    p.add(name, *components, mode="lines", **kwargs)

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
        col_vecs[:,i] /= lengths[i]
        # Orthgonolize, remove vector i from all remaining vectors.
        if (i+1 < col_vecs.shape[1]):
            v = np.dot(col_vecs[:,i],col_vecs[:,i+1:]) * col_vecs[:,i:i+1]
            col_vecs[:,i+1:] -= v
    # Return the orthonormalized vectors and their lengths (before normalization).
    return col_vecs, lengths

# Compute the singular values and the right singular vectors for a matrix of row vectors.
def svd(row_vecs, steps=1, bias=1.0):
    dim = row_vecs.shape[1]
    # Initialize a holder for the singular valeus and the right
    #   singular vectors (the principal components).
    # Rescale the data for stability control, exit early when given only zeros.
    multiplier = abs(row_vecs).max()
    assert (multiplier > 0), "ERROR: Provided 'row_vecs' had no nonzero entries."
    multiplier = bias / multiplier
    # Compute the covariance matrix (usually the most expensive operation).
    covariance = np.matmul(multiplier*row_vecs.T, multiplier*row_vecs)
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


# --------------------------------------------------------------------
#                     GENERATE SAMPLE DATA
print('-'*70)
# Configurations.
np.random.seed(2)
n = 100000
d = 64
l = 1
i = np.linspace(0,d-l,min(20,d-l)).round().astype(int)
eps = 2**(-26)
variances = np.clip(np.linspace(-.2, 1.8, d), 0, float('inf'))

# Generate random data.
x = np.random.normal(0.0, variances, size=(n,d))
# Apply a random transformation to "x" to make the problem harder.
from tlux.random import sphere
x = x @ sphere(d,d)
print(f"Done allocating {x.shape} test data")
print(" with", sum((variances < eps).astype(int)), "zero singular values.")
print()


# --------------------------------------------------------------------
# Custom Python SVD call.
temp = x - x.mean(axis=0)
start = time.time()
v, s = svd(temp, steps=5, bias=1/8)
end = time.time()
print()
print(f" {end - start:.2f} seconds (custom Python SVD)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()


# --------------------------------------------------------------------
# Fortran custom SVD call.
from tlux.math import svd as fast_svd
temp = np.asarray((x - x.mean(axis=0)), order='C', dtype="float32")
start = time.time()
s, vt, rank = fast_svd(temp.T, steps=10, bias=1/8)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Fortran SVD)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()

temp = np.asarray((x - x.mean(axis=0)), order='F', dtype="float32")
start = time.time()
s, vt, rank = fast_svd(temp, bias=1/8)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Fortran SVD)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()


# --------------------------------------------------------------------
# Python numpy SVD.
temp = x - x.mean(axis=0)
start = time.time()
s = np.linalg.svd(temp, compute_uv=False)
end = time.time()
print()
print(f" {end - start:.2f} seconds (numpy SVD)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()


# --------------------------------------------------------------------
# Fortran LAPACK SVD call (only singular values).
import fmodpy
LAPACK = fmodpy.fimport("lapack.f90", lapack=True, verbose=False)
temp = np.asarray((x - x.mean(axis=0)).T, order='F', dtype="float32")
start = time.time()
_, s = LAPACK.lapack_sing_vals(temp)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Fortran LAPAPCK)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()


# --------------------------------------------------------------------
# Use "sklearn" to compute the principal compoents 
def pca(x, num_components=None):
    from sklearn.decomposition import PCA
    if (num_components is None): num_components = min(*x.shape)
    else: num_components = min(num_components, *x.shape)
    pca = PCA(n_components=num_components)
    pca.fit(x)
    return pca.components_, pca.singular_values_
# Python sklearn PCA call.
start = time.time()
vecs, vals = pca(x)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Python sklearn)", end="")
print(" with", sum((vals < eps).astype(int)), "zeros")
print(vals[i])
