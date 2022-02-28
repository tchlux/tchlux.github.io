import time
import numpy as np


# Use "sklearn" to compute the principal compoents 
def pca(x, num_components=None):
    from sklearn.decomposition import PCA
    if (num_components is None): num_components = min(*x.shape)
    else: num_components = min(num_components, *x.shape)
    pca = PCA(n_components=num_components)
    pca.fit(x)
    return pca.components_, pca.singular_values_


# Given a plot object and a series name, add a bunch of lines for "row_vecs".
def plot_vecs(p, name, row_vecs, **kwargs):
    # Break into components.
    components = [
        sum(([0,v,None] for v in row_vecs[:,i]),[])
        for i in range(row_vecs.shape[1])
    ]
    p.add(name, *components, mode="lines", **kwargs)


# --------------------------------------------------------------------
#                     GENERATE SAMPLE DATA
np.random.seed(2)
print('-'*70)
# n = 1000
# d = 2**8
n = 3
d = 4
variances = np.clip(np.linspace(-.2, 1.8, d), 0, float('inf'))
x = np.random.normal(0.0, variances, size=(n,d))
# x = x.T
n, d = x.shape
i = np.linspace(0,min(n,d)-1,20).round().astype(int)
print(f"Done allocating {x.shape} test data")
print(" with", sum((variances < 2**(-26)).astype(int)), "zero singular values.")
print()
eps = 2**(-13)


# Fortran custom SVD call.
from tlux.math import svd, orthogonalize
temp = np.asarray((x - x.mean(axis=0)), order='F', dtype="float32")
start = time.time()
if (n >= d):
    u, s, vt = svd(temp, u=True, vt=True, steps=0, kmax=10)
else:
    vt, s, ut = svd(temp, u=True, vt=True, steps=0, kmax=10)
    u = ut.T
end = time.time()
print()
print(f" {end - start:.2f} seconds (Fortran SVD)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()
print("u.shape: ", u.shape)
print("s.shape: ", s.shape)
print("vt.shape:", vt.shape)
print("u.flatten().round(3):  ", u.round(3))
print("s.flatten().round(3):  ", s.round(3))
print("vt.flatten().round(3): ", vt.round(3))
print("u u.T", np.matmul(u, u.T))
print("v v.T", np.matmul(vt.T, vt))
print()
print(temp)
print(u @ np.diag(s) @ vt.T)
print()
print(temp.T)
print(vt @ np.diag(s) @ u.T)
print()

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
u, s, vt = np.linalg.svd(temp)
print("u.shape: ", u.shape)
print("s.shape: ", s.shape)
print("vt.shape:", vt.shape)
print("u.flatten().round(3):  ", u.round(3))
print("s.flatten().round(3):  ", s.round(3))
print("vt.flatten().round(3): ", vt.round(3))
print("u u.T", np.matmul(u, u.T))
print("v v.T", np.matmul(vt.T, vt))
print()
exit()

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


# Python sklearn PCA call.
start = time.time()
vecs, vals = pca(x)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Python sklearn)", end="")
print(" with", sum((vals < eps).astype(int)), "zeros")
print(vals[i])
