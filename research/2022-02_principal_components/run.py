import time
import numpy as np
from tlux.random import sphere

# Generate random normal points.
def normal(n, d):
    return np.random.normal(size=(n,d))


# Configure the model.
input_dim = 2
output_dim = 1
state_dim = 32
num_states = 8

# Initialize the weights (input, internal, output).
weight_matrices = [sphere(input_dim, state_dim)] + \
                  [sphere(state_dim, state_dim)
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
        # Normalize then orthogonalize, remove vector i from all remaining vectors.
        if (i+1 < col_vecs.shape[1]):
            v = np.dot(col_vecs[:,i],col_vecs[:,i+1:]) * col_vecs[:,i:i+1]
            col_vecs[:,i+1:] -= v
    # Return the orthonormalized vectors and their lengths (before normalization).
    return col_vecs, lengths


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



np.set_printoptions(linewidth=1000, formatter=dict(float=lambda s: f"{s: .2e}"))
for i in range(num_states):
    vecs, vals = pca(states[i,:,:])
    print(f"{i:3d}", vals.round(3))
    # print(np.matmul(vecs, vecs.T))
    vecs, vals = svd(states[i,:,:])
    print(f"   ", vals.round(3))
    # print(np.matmul(vecs, vecs.T))
    print()

