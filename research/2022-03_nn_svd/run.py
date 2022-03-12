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

auto_show = False # Change to "True" to automatically open in browser.

# --------------------------------------------------------------------
# Random normal weight distributions.
lengths = np.linalg.norm(normal(10000, 3), axis=1)
p = Plot("", "2-norm length", "count")
p.add_histogram("lengths", lengths, color=1, num_bins=60)
p.plot(file_name=f"normal_init/lengths.html", show=auto_show, show_legend=False)
# --------------------------------------------------------------------
exit()
# Looking at principal components of data.

# --------------------------------------------------------------------
# Use "sklearn" to compute the principal compoents and project data down.
def project(x, num_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)
    pca.fit(x)
    return np.matmul(x, pca.components_.T)

# Generate a projection and a visual for the input data.
x_3d = project(x, 3)
p = Plot()
p.add("Data", *x_3d.T, marker_size=5, marker_line_width=2, color=1, shade=True)
p.plot(file_name="input_data.html", show=auto_show, show_legend=False)

# Cycle through and make visuals for all internal layers.
for i in (0, num_states//2, num_states-1):
    x_3d = project(states[i,:,:], 3)
    p = Plot()
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
    x_3d = project(states[i,:,:], 3)
    p = Plot()
    p.add("Data", *x_3d.T, marker_size=5, marker_line_width=2, color=1, shade=True)
    p.plot(file_name=f"sphere_init/data_layer_{i+1}.html", show=auto_show, show_legend=False)
# --------------------------------------------------------------------

# Use "sklearn" to get the singular values of data.
def singular_values(x):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(x.shape))
    pca.fit(x)
    return pca.singular_values_

# Raise the input dimension of the model and data.
input_dim = 40
weight_matrices[0] = sphere(input_dim, state_dim)
x = ball(1000, input_dim)
y = np.cos(np.linalg.norm(x, axis=1, keepdims=True))
states = np.zeros((num_states, x.shape[0], state_dim))
forward(x, states=states)

p = Plot("", "principal component", "singular value")
for i in range(num_states):
    sv = singular_values(states[i,:,:])
    p.add(f"layer {i}",
          sum(([j,j,None] for j in range(len(sv))),[]),
          sum(([0,sv[j],None] for j in range(len(sv))),[]),
          mode="lines",
          line_width=10,
          marker_line_width=1,
          color=i,
          frame=i+1,
    )
p.show(file_name="sphere_init/singular_values.html",
       show_legend=False, bounce=True, frame_label="Layer ")
