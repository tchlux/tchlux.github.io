import time
import numpy as np
from tlux.random import well_spaced_box as box
from tlux.random import well_spaced_sphere as sphere
from tlux.random import well_spaced_ball as ball
from tlux.plot import Plot

np.random.seed(0)

# Generate random normal points.
def normal(n, d):
    return np.random.normal(size=(n,d))

# --------------------------------------------------------------------
# Configure the model.
input_dim = 3
output_dim = 1
state_dim = 40
num_states = 10

# Initialize the weights (input, internal, output).
weight_matrices = [normal(input_dim, state_dim)] + \
                  [normal(state_dim, state_dim)
                   for i in range(num_states-1)] + \
                  [np.zeros((state_dim, output_dim))]
shift_vectors = [
    np.linspace(-1, 1, state_dim) for i in range(num_states)
]

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
x = ball(100, input_dim)
y = np.cos(np.linalg.norm(x, axis=1, keepdims=True))

# Initialize holder for the states.
states = np.zeros((num_states, x.shape[0], state_dim))
forward(x, states=states)

print()
print(x.shape)
print(y.shape)
print(forward(x).shape)
print(states.shape)
# --------------------------------------------------------------------

# Generating visuals.

auto_show = True # Change to "True" to automatically open in browser.

# --------------------------------------------------------------------
# Random normal weight distributions.
lengths = np.linalg.norm(normal(1000000, 3), axis=1)
p = Plot(f"Distribution of 1M normal weight vector 2-norm lengths")
p.add_histogram("lengths", lengths, color=1, num_bins=200)
p.plot(file_name=f"normal_init/lengths.html", show=auto_show, show_legend=False)
# --------------------------------------------------------------------

# Looking at principal components of data.

# --------------------------------------------------------------------
# Use "sklearn" to compute the principal compoents 
def pca(x, num_components=None):
    from sklearn.decomposition import PCA
    if (num_components is None): num_components = min(*x.shape)
    else: num_components = min(num_components, *x.shape)
    pca = PCA(n_components=num_components)
    pca.fit(x)
    return pca.components_, pca.singular_values_

# Generate a projection and a visual for the input data.
projection, _ = pca(x, num_components=3)
x_3d = np.matmul(x, projection.T)
p = Plot("Data distribution in first 3 principal components at initialization.")
p.add("Data", *x_3d.T, marker_size=5, marker_line_width=2, color=1, shade=True)
p.plot(file_name="input_data.html", show=auto_show, show_legend=False)

# Cycle through and make visuals for all internal layers.
for i in (0, num_states//2, num_states-1):
    projection, _ = pca(states[i,:,:], num_components=3)
    x_3d = np.matmul(states[i,:,:], projection.T)
    p = Plot(f"Layer {i+1} data distribution (first 3 principal components)")
    p.add("Data", *x_3d.T, marker_size=5, marker_line_width=2, color=1, shade=True)
    p.plot(file_name=f"normal_init/data_layer_{i+1}.html", show=auto_show, show_legend=False)

# Initialize the weights with a sphere method instead.
weight_matrices = [sphere(input_dim, state_dim)] + \
                  [sphere(state_dim, state_dim)
                   for i in range(num_states-1)] + \
                  [np.zeros((state_dim, output_dim))]
forward(x, states=states)

# Cycle through and make visuals for all internal layers.
for i in (0, num_states//2, num_states-1):
    projection, _ = pca(states[i,:,:], num_components=3)
    x_3d = np.matmul(states[i,:,:], projection.T)
    p = Plot(f"Layer {i+1} data distribution (first 3 principal components)")
    p.add("Data", *x_3d.T, marker_size=5, marker_line_width=2, color=1, shade=True)
    p.plot(file_name=f"sphere_init/data_layer_{i+1}.html", show=auto_show, show_legend=False)
# --------------------------------------------------------------------
