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
d = 3
variances = np.clip(np.linspace(-.2, 1.8, d), 0, float('inf'))
x = np.random.normal(0.0, variances, size=(n,d))
# x = x.T
n, d = x.shape
i = np.linspace(0,min(n,d)-1,min(20,max(n,d))).round().astype(int)
print(f"Done allocating {x.shape} test data")
print(" with", sum((variances < 2**(-26)).astype(int)), "zero singular values.")
print()
eps = 2**(-13)


# Fortran custom SVD call.
from tlux.math import svd

temp = np.asarray((x - x.mean(axis=0)), order='F', dtype="float32")
start = time.time()
if (n >= d):
    u, s, vt = svd(temp, u=True, vt=True, steps=100, kmax=10)
else:
    vt, s, ut = svd(temp, u=True, vt=True, steps=100, kmax=10)
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
print("u.flatten().round(3):  ")
print(u.round(3))
print("s.flatten().round(3):  ")
print(s.round(3))
print("vt.flatten().round(3): ")
print(vt.round(3))
print()
print("u u.T")
print(np.matmul(u, u.T))
print("u.T u")
print(np.matmul(u.T, u))
print("v v.T")
print(np.matmul(vt.T, vt))
print("v.T v")
print(np.matmul(vt, vt.T))
print()
print("Max error: (normal)")
print("", abs(temp - u @ np.diag(s) @ vt.T).max())
print()
print("Max error: (transpose)")
print("", abs(temp - u @ np.diag(s) @ vt.T).max())
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
print("u.flatten().round(3):  ")
print(u.round(3))
print("s.flatten().round(3):  ")
print(s.round(3))
print("vt.flatten().round(3): ")
print(vt.round(3))
print()
exit()

# Fortran "CMLIB" call that uses the bidiagonalization method.
def cmlib(A):
    import fmodpy
    _cmlib = fmodpy.fimport("cmlib.f", end_is_named=False, f_compiler_args="--std=legacy -fPIC -shared -O3")
    A = np.asarray(A, order="F", dtype="float32")
    NM = A.shape[0]
    M = A.shape[0]
    N = A.shape[1]
    MATU = False
    MATV = True
    W = np.zeros(N, dtype="float32")
    U = A
    V = A
    IERR = 0
    RV1 = np.zeros(N, dtype="float32")
    NM, M, N, A, W, MATU, U, MATV, V, IERR, RV1 = (
        _cmlib.svd(NM, M, N, A, W, MATU, U, MATV, V, IERR, RV1)
    )
    s = W
    V = A[:N,:N]
    return s, V.T
temp = np.asarray((x - x.mean(axis=0)), order='C', dtype="float32")
start = time.time()
s, vt = cmlib(temp)
end = time.time()
print()
print(f" {end - start:.2f} seconds (Fortran cmlib)", end="")
print(" with", sum((s < eps).astype(int)), "zeros")
print(s[i])
print()

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


# 2022-03-01 07:55:20
# 
##########################################################################################################################################
# from tlux.math import orthogonalize
# a = [-0.369071245, 0.815281034, -0.446209908,  0.728322029,  -4.45361324E-02, -0.683786213,   0.00000000,   0.00000000, 0.00000000 ] # #
# a = np.asarray(a).reshape((3,3)).astype("float32")                                                                                   # #
# print("A:")                                                                                                                          # #
# print(a.T)                                                                                                                           # #
# print()                                                                                                                              # #
# # a = np.asarray(                                                                                                                    # #
# #     [[-0.369,  0.815, -0.446],                                                                                                     # #
# #      [ 0.728, -0.045, -0.684],                                                                                                     # #
# #      [ 0.,     0.,     0.   ]]                                                                                                     # #
# # )                                                                                                                                  # #
# r = np.zeros(max(a.shape), dtype="float32")                                                                                            #
# q, r, rank = orthogonalize(a.T, r, rank=0)                                                                                             #
# print("rank: ",rank)                                                                                                                   #
# a.T[:,:rank], _, rank = orthogonalize(a[:rank].T, np.zeros(r.size,dtype=r.dtype), rank=0, trans=True)                                  #
# print("rank: ",rank)                                                                                                                   #
# print()                                                                                                                                #
# print("Q:")                                                                                                                          # #
# print(q)                                                                                                                             # #
# print()                                                                                                                              # #
# print("R (diagonal)")                                                                                                                # #
# print(r)                                                                                                                             # #
# print()                                                                                                                              # #
# print("Qt Q")                                                                                                                        # #
# print(q.T @ q)                                                                                                                       # #
# print()                                                                                                                              # #
# print("Q Qt")                                                                                                                        # #
# print((q) @ (q.T))                                                                                                                     #
# print()                                                                                                                              # #
# print(np.linalg.qr(a.T)[0])                                                                                                          # #
# exit()                                                                                                                               # #
##########################################################################################################################################


  # CONTAINS
  #   ! Order the column vectors in Q by their magnitdues.
  #   SUBROUTINE REORDER(Q, LENGTHS)
  #     REAL(KIND=RT), DIMENSION(:,:) :: Q
  #     REAL(KIND=RT), DIMENSION(:) :: LENGTHS
  #     ! Local variables.
  #     INTEGER,       DIMENSION(SIZE(Q,2)) :: LI
  #     REAL(KIND=RT), DIMENSION(SIZE(Q,2)) :: LV
  #     INTEGER :: I
  #     REAL(KIND=RT) :: VAL, VEC(SIZE(Q,1))
  #     ! Initialize all the indices and (negated) values.
  #     FORALL (I=1:SIZE(Q,2)) LI(I) = I
  #     LV(:) = -LENGTHS(:)
  #     ! Sort the values (tracking corresponding original indices).
  #     CALL ARGSORT_REAL(LV(:), LI(:))
  #     ! Reorder Q and LENGTHS accordingly.
  #     DO I = 1, SIZE(Q,2)
  #        IF ((LI(I) .NE. I) .AND. (LV(I) .NE. 0.0_RT)) THEN
  #           ! Swap the column vectors in Q.
  #           VEC(:) = Q(:,I)
  #           Q(:,I) = Q(:,LI(I))
  #           Q(:,LI(I)) = VEC(:)
  #           ! Swap the length terms in S.
  #           VAL = LENGTHS(I)
  #           LENGTHS(I) = LENGTHS(LI(I))
  #           LENGTHS(LI(I)) = VAL
  #        END IF
  #     END DO
  #   END SUBROUTINE REORDER
  # END SUBROUTINE SVD
