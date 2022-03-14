import numpy as np

# Create weight matrices and shift vectors for new model.
def new_model(state_dims, seed=0):
    np.random.seed(seed)
    weights = [
        np.random.uniform(size=(dim_in, dim_out)) for (dim_in, dim_out)
        in zip(state_dims[:-1], state_dims[1:])
    ]
    shifts = [
        np.random.uniform(size=(dim,)) for dim in state_dims[1:-1]
    ]
    return weights, shifts

# Define the forward function for the ReLU MLP.
def evaluate(x, weights, shifts, states=None):
    x = x.reshape((-1, weights[0].shape[0]))
    if (states is not None): states[0] = x[:,:]
    for i in range(len(weights)-1):
        x = np.clip(np.matmul(x, weights[i]) + shifts[i], 0.0, float("inf"))
        if (states is not None): states[i+1] = x[:,:]
    return np.matmul(x, weights[-1])

# Define a "gradient" function that computes the mean squared error
# gradient of weights and shifts when approximating "y".
def gradient(x, y, weights, shifts):
    # Run the model forward.
    states = [None] * len(weights)
    output = evaluate(x, weights, shifts, states=states)
    # Compute the squared error gradient.
    grad = (output - y)
    # Store the mean squared error.
    mean_squared_error = np.mean(grad**2)
    # Compute the gradient for all weights and shifts.
    weights_grad = [None] * len(weights)
    shifts_grad = [None] * len(shifts)
    weights_grad[-1] = np.matmul(states[-1].T, grad)
    for i in range(len(weights)-2, -1, -1):
        grad = np.where(states[i+1] > 0, np.matmul(grad, weights[i+1].T), 0.0)
        shifts_grad[i] = np.mean(grad, axis=0)
        weights_grad[i] = np.matmul(states[i].T, grad)
    # Return the gradient.
    return weights_grad, shifts_grad, mean_squared_error

# Take one mean squared error gradient step for a model.
def step(x, y, weights, shifts, step_size=0.0001):
    weights_grad, shifts_grad, mse = gradient(x, y, weights, shifts)
    for i in range(len(weights)):
        weights[i] -= step_size * weights_grad[i]
    for i in range(len(shifts)):
        shifts[i] -= step_size * shifts_grad[i]
    return mse


# Configure the model.
input_dim = 2
output_dim = 1
state_dim = 100
num_states = 1
num_steps = 10001
num_printouts = 31

# Initialize a model.
state_dims = (input_dim,) + (state_dim,) * num_states + (output_dim,)
weights, shifts = new_model(state_dims)

# Define some test data.
n = 200
x = (np.random.uniform(size=(n, input_dim)) - 0.5)
y = 1 - np.cos(5*np.linalg.norm(x, axis=1, keepdims=True))

# Set the print interval.
print_at = set(np.linspace(0, num_steps-1, num_printouts).round())

x_min_max = np.asarray([np.min(x, axis=0), np.max(x, axis=0)]).T
# Train the model.
for i in range(num_steps):
    mse = step(x, y, weights, shifts)
    if (i in print_at):
        print(f"{i:5d}: ", mse)
