import time
import numpy as np

# Configure the model.
input_dim = 2
output_dim = 1
state_dim = 32
num_states = 16

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

# Define some random data.
x = np.random.random(size=(100, input_dim))
y = np.cos(np.linalg.norm(x, axis=1, keepdims=True))

# Initialize holder for the states.
states = np.zeros((num_states, x.shape[0], state_dim))
forward(x, states=states)

print()
print(x.shape)
print(y.shape)
print(forward(x).shape)
print(states.shape)


# Use "sklearn" to compute the principal compoents 
def pca(x, num_components=None):
    from sklearn.decomposition import PCA
    if (num_components is None): num_components = min(*x.shape)
    else: num_components = min(num_components, *x.shape)
    pca = PCA(n_components=num_components)
    pca.fit(x)
    return pca.components_, pca.singular_values_


# Use an SVD to get the principal components of a set of row vectors.
def pca(row_vecs):
    from tlux.math import svd, orthogonalize
    vecs = np.array(row_vecs, order='F', dtype="float32")
    u, s, vt = svd(vecs, u=True, vt=True, steps=10000)
    if (s.size < vt.shape[0]):
        s = np.concatenate((s, np.zeros(vt.shape[0] - s.size)))
    v = vt.T
    print()
    print("vecs.shape: ",vecs.shape)
    print("u.shape: ",u.shape)
    print("v.shape: ",v.shape)
    print("s.shape: ",s.shape)
    print(s)
    print(vecs - u[:,:] @ np.diag(s[:]) @ v[:,:])
    # A  = U s Vt
    # At = V s Ut
    # At U
    # Vt At / s = Ut
    # (m x n) = (m x n) (n) (n x n)
    # 
    # At A = Vt s Ut U s V
    #      = Vt s2 V
    #      = s2 V
    # U = (A Vt) / s
    # 
    # (m x n) = ((m x n) (n x n)) / (n)f
    exit()
    return v.T, s

print()
x = np.random.random(size=(4,3))
pca(x)


np.set_printoptions(linewidth=1000, formatter=dict(float=lambda s: f"{s: .2e}"))
for i in range(num_states):
    vecs, vals = pca(states[i,:,:])
    print(f"{i:3d}", vals.round(3))
    print(np.matmul(vecs, vecs.T))
    vecs, vals = pca(states[i,:,:].T)
    print(f"   ", vals.round(3))
    print(np.matmul(vecs, vecs.T))
    print()

# TODO: 
#  - Figure out how to measure the 2-norm of the PCA projection.
#  - 
