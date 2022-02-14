print("Importing modules..")
import time
import logging
from typing import Optional, List, Tuple
import numpy as np
import torch
print(" done.")
print()


MODEL_LAYERS = (32,) * 6
MODEL_TRAINING_STEPS = 3000
MODEL_TRAINING_PRINTOUTS = 20
SHOW_TRAINING_LOGS = True

logging.getLogger().setLevel(logging.INFO) # Show logs for information.
# logging.getLogger().setLevel(logging.WARNING) # Only show logs for warnings & errors.


# Make the ROW VECTORS in x orthogonal and unit 2-norm.
def orthonormalize(x: np.ndarray) -> np.ndarray:
    x[0] /= np.linalg.norm(x[0])
    for i in range(1,x.shape[0]):
        x[i:] = (x[i:].T - np.dot(x[i:], x[i-1]) * x[i-1][:,None]).T
        x[i] /= np.linalg.norm(x[i])
    return x


# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_ball(
    num_points: int,
    dimension: int,
    radius: float=1.0,
    inside: bool=True,
    ortho: bool=False
) -> np.ndarray:
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Orthogonalize the first "dimension" directions if requested.
    if (ortho):
        orthonormalize(random_directions[:,:dimension].T)
    # Place some points inside the ball (uniform density) if requested.
    if (inside):
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = random.random(num_points) ** (1/dimension)
        # Return the list of random (direction & length) points.
        return radius * (random_directions * random_radii).T
    # Otherwise return the points that are randomly placed on the sphere.
    else:
        return random_directions.T

# Construct normalization constants for making data zero mean and unit variance.
#   Optionally, apply the same normalization to some validation data.
def normalize_data(
    x: List[List[float]],
    y: List[List[float]],
    xv: Optional[List[List[float]]] = None,
    yv: Optional[List[List[float]]] = None,
    logs: bool = SHOW_TRAINING_LOGS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            Optional[torch.Tensor], Optional[torch.Tensor]]:
    if logs:
        logging.info("  Normalizing data..")
    # Convert the data into floats and do some preprocessing.
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype="float32")
    # Compute shift and scale for inputs to make them zero mean and unit variance (handle NaNs by mean-substitution).
    x_mean = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), x_mean, x)
    x_var = (x - x_mean).var(axis=0)
    x_var = np.where(x_var > 0, x_var, 1)
    # Normalize the outputs for training to make loss function consistent.
    y_mean = np.nanmean(y, axis=0)
    y = np.where(np.isnan(y), y_mean, y)
    y_var = y.var(axis=0)
    y_var = np.where(y_var > 0, y_var, 1)
    # Provide a summary of training data (for sanity checking).
    if logs:
        logging.info("  Summary of data before normalization:")
        logging.info(f"   x mean: {x_mean.mean()}")
        logging.info(f"      var: {x_var.mean()}")
        logging.info(f"      min: {x.min()}")
        logging.info(f"      max: {x.max()}")
        logging.info(f"   y mean: {[round(_,2) for _ in y_mean]}")
        logging.info(f"      var: {[round(_,2) for _ in y_var]}")
        logging.info(f"      min: {[round(_,1) for _ in y.min(axis=0)]}")
        logging.info(f"      max: {[round(_,1) for _ in y.max(axis=0)]}")
        logging.info("")
        logging.info("  Converting data to Tensor (and normalizing) for training..")
    # Normalize the data so that it doesn't have to be done during training (for every step).
    x = (x - x_mean) / x_var
    y = (y - y_mean) / y_var
    # Normalize the validation data as well (if it was provided).
    if xv is not None:
        xv = np.asarray(xv, dtype="float32")
        xv = np.where(np.isnan(xv), x_mean, xv)
        xv = (xv - x_mean) / x_var
        xv = torch.as_tensor(xv)
    if yv is not None:
        yv = np.asarray(yv, dtype="float32")
        yv = np.where(np.isnan(yv), y_mean, yv)
        yv = (yv - y_mean) / y_var
        yv = torch.as_tensor(yv)
    # Convert shift and scale into PyTorch float tensors.
    return (
        torch.as_tensor(x),
        torch.as_tensor(y),
        torch.as_tensor(x_mean),
        torch.as_tensor(x_var),
        torch.as_tensor(y_mean),
        torch.as_tensor(y_var),
        xv,
        yv,
    )


# Given data, optimize the paramters of this model to fit the data.
def optimize_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    xv: torch.Tensor,
    yv: torch.Tensor,
    parameters : List[torch.Tensor],
    training_loss : torch.Tensor,
    validation_loss : torch.Tensor,
    step_sizes : torch.Tensor,
    num_steps: int,
    logs: bool,
    print_frequency: int,
) -> None:
    # Initialize the optimization parameters
    #   (torchscript doesn't allow use of globals here).
    prev_loss = float('inf')
    step_size   = 0.01
    mean_change = 0.1
    mean_remain = 1.0 - mean_change
    curv_change = 0.01
    curv_remain = 1.0 - curv_change
    faster_rate = 1.01
    slower_rate = 0.99
    # Store the model with the best loss.
    best_loss = float('inf')
    best_params = [p.data.clone() for p in parameters]
    # Initialize mean and curvature estimates for optimization.
    means = [torch.zeros(p.shape) for p in parameters]
    curvs = [torch.ones(p.shape) for p in parameters]
    has_validation = (xv is not None) and (yv is not None)
    # Iterate over the predetermined number of steps.
    for s in range(num_steps):
        # Determine whether or not logs will be shown in this step.
        show_log = logs and ((s % print_frequency) == 0)
        # Zero out the gradient of all parameters (preparing for computations).
        for p in parameters:
            p.grad.zero_()
            # if p.grad is not None:
        # Run the model forward and compute the loss.
        loss = ((model.forward(x) - y)**2).mean()
        training_loss[s] = loss
        # Pass the loss gradient backwards through the model.
        loss.backward()
        # Initialize a container for storing validation loss.
        vloss = loss 
        # Update the model parameters based on the gradient.
        with torch.no_grad():
            # Print updates if applicable.
            info = ""
            # Compute the validation loss.
            if (has_validation):
                validation_predictions = model.forward(xv)
                vloss = ((validation_predictions - yv)**2).mean()
                validation_loss[s] = vloss
                if show_log:
                    info = "  Step " + str(s) + " loss: " + str(loss.item())
                    info += "   (validation loss: " + str(vloss.item()) + ")"
                    logging.info(info)
            elif show_log:
                info = "  Step " + str(s) + " loss: " + str(loss.item())
                logging.info(info)
            # Store the parameters that perform the best.
            if (has_validation):
                if (vloss < best_loss):
                    for i,p in enumerate(parameters):
                        best_params[i].copy_(p.data)
                    best_loss = vloss
            else:
                if (loss < best_loss):
                    for i,p in enumerate(parameters):
                        best_params[i].copy_(p.data)
                    best_loss = loss
            # Update stepping parameters based on loss change.
            if (loss < prev_loss):
                step_size *= faster_rate
                mean_change *= slower_rate
                mean_remain = 1.0 - mean_change
                curv_change *= slower_rate
                curv_remain = 1.0 - curv_change
            else:
                step_size *= slower_rate
                mean_change *= faster_rate
                mean_remain = 1.0 - mean_change
                curv_change *= faster_rate
                curv_remain = 1.0 - curv_change
            prev_loss = float(loss.item())
            # Take the gradient descent step.
            step_magnitude = 0.0
            for i,p in enumerate(parameters):
                g = p.grad
                means[i] = mean_remain * means[i] + mean_change * g
                curvs[i] = curv_remain * curvs[i] + curv_change * (g - means[i])**2
                step = step_size * means[i] / torch.sqrt(curvs[i])
                p -= step
                step_magnitude += (step**2).sum()
            # Record the step size.
            step_sizes[s] = torch.sqrt(step_magnitude)
    # Print out the final training and testing loss.
    with torch.no_grad():
        if logs:
            if (has_validation):
                info = "  Step " + str(num_steps) + " loss: " + str(best_loss.item())
            else:
                info = "  Step " + str(num_steps) + " validation loss: " + str(best_loss.item())
            logging.info(info)
        # Revert to the best parameters that were observed.
        for p_now, p_best in zip(parameters, best_params):
            p_now.copy_(p_best)


# Given x (input) and y (output) data for training, fit a model to the data
#  and store normalization constants internally for future application. Uses
#  zero mean and unit variance for both inputs and outputs. If optional validation
#  data is provided, loss information during validation is printed through training.
def fit(
    model: torch.nn.Module,
    x: List[List[float]],
    y: List[List[float]],
    xv: Optional[List[List[float]]] = None,
    yv: Optional[List[List[float]]] = None,
    steps: int = MODEL_TRAINING_STEPS,
    printouts: int = MODEL_TRAINING_PRINTOUTS,
    logs: bool = SHOW_TRAINING_LOGS,
) -> torch.nn.Module:
    # Normalize the data to have zero mean and unit variance.
    (
        x,
        y,
        model.input_shift,
        model.input_scale,
        model.output_shift,
        model.output_scale,
        xv,
        yv,
    ) = normalize_data(x, y, xv, yv, logs)
    # Initialize a new model if one was not provided.
    if not model.initialized:
        if logs:
            logging.info("  Initializing model..")
        model.initialize(x.shape[1], y.shape[1])
    # Switch the model to training mode.
    model.train()
    print_frequency = (steps + printouts - 1) // printouts
    if logs:
        logging.info("  Training model..")
    # Run a single point forwards and backwards to initialize gradients.
    model.forward(x[:3]).sum().backward()
    # Train this model (with a compiled optimizer).
    parameters = list(model.parameters())
    training_loss = torch.zeros(steps, dtype=torch.float32)
    validation_loss = torch.zeros(steps, dtype=torch.float32)
    step_sizes = torch.zeros(steps, dtype=torch.float32)
    # Run the optimization.
    optimize_model(
        model,
        x, y, xv, yv, parameters, training_loss, validation_loss,
        step_sizes, steps, logs, print_frequency,            
    )
    # Switch the model to evaluation mode.
    model.eval()
    # Store the information about training.
    model.training_info = {
        "loss_function": "Mean-Squared-Error",
        "optimizer": "Adaptive-Adam",
        "optimizer_steps": steps,
        "training_loss": training_loss.detach().numpy(),
        "validation_loss": validation_loss.detach().numpy(),
        "step_sizes": step_sizes.detach().numpy(),
    }
    return model


# A multilayer perceptron (MLP) designed to do data normalization internally at runtime
#  that also provides a convenience method for producing embedded representations.
class MLP(torch.nn.Module):
    # Initialize an MLP optionally provide the number of components in an input vector,
    #  the number of components in an output vector, the shape of the MLP in terms of
    #  number of hidden nodes per layer (the size of the last internal layer 
    #  is referred to here as the "embedding").
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        layers: Tuple[int] = MODEL_LAYERS,
    ) -> None:
        super().__init__()
        self.embedder_shape = layers[:-1]
        self.embedding_size = layers[-1]
        self.embedder = None
        self.last_layer = None
        self.training_info = None
        self.initialized = False
        if (input_dim is not None) and (output_dim is not None):
            self.initialize(input_dim, output_dim)
    
    def initialize(self, input_dim: int, output_dim: int):
        # Construct a tuple that holds the sizes of all layers in this model.
        layers = (input_dim,) + self.embedder_shape
        # Create an embedder (the main body of the model that takes input).
        self.embedder = torch.nn.Sequential(
            *(sum(([torch.nn.Linear(in_size, out_size), torch.nn.ReLU()]
                   for (in_size, out_size) in zip(layers[:-1], layers[1:])), [])
              + [torch.nn.Linear(layers[-1], self.embedding_size)]
        ))
        # Make all the vectors unit length and equidistributed on the 
        #  sphere, and make all the bias terms uniformly distributed.
        for i,p in enumerate(self.embedder.parameters()):
            if (len(p.data.shape) == 2):
                vecs = random_ball(*p.data.shape, ortho=True)
                p.data.copy_(torch.as_tensor(vecs))
            else:
                p.data.copy_(torch.as_tensor(np.linspace(-2, 2, p.data.shape[0])))
        # Create a linear output layer (if output size is known), otherwise insert a temporary function.
        self.last_layer = torch.nn.Linear(self.embedding_size, output_dim)
        # Make all the vectors unit length and equidistributed on the 
        #  sphere, and make all the bias terms uniformly distributed.
        for i,p in enumerate(self.last_layer.parameters()):
            if (len(p.data.shape) == 2):
                vecs = random_ball(*p.data.shape, ortho=True)
                orthonormalize(vecs[:vecs.shape[1]])
                p.data.copy_(torch.as_tensor(vecs))
            else:
                p.data.copy_(torch.as_tensor(np.linspace(-.2, .2, p.data.shape[0])))
        # Record that this model has been initialized.
        self.initialized = True

    # Wrapper for the 'fit' function over this model.
    def fit(self, *args, **kwargs):
        return fit(self, *args, **kwargs)

    # Pass data through this network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_layer(self.embedder(x))

    # Save this model to a path.
    def save(self, path: str) -> None:
        torch.save(self, path)

    # Given some unnormalized data, embed it into the last-layer
    #  representation. Return results in a numpy matrix.
    def embed(self, x: List[List[float]]) -> List[List[float]]:
        with torch.no_grad():
            x_normalized = (x - self.input_shift) / self.input_scale
            y_embedded = self.embedder(x_normalized)
        return y_embedded.detach().numpy()

    # Normalize inputs, then pass them forward through the network.
    #   Disable gradients for speed. Return Numpy array as output.
    def predict(self, x: List[List[float]]) -> List[List[float]]:
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x_normalized = (x - self.input_shift) / self.input_scale
            y_normalized = self.forward(x_normalized)
            y = y_normalized * self.output_scale + self.output_shift
        return y.detach().numpy()

    # When called with "model(<input>)" assume unnormalized data.
    def __call__(self, *args, **kwargs) -> List[List[float]]:
        return self.predict(*args, **kwargs)


# Create a class that Aggregates input features in one MLP, then exchanges them in another MLP.
class AggExc(torch.nn.Module):
    # Initialize two MLPs for this.
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        representation_dim: Optional[int] = None,
        location_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.aggregator = None
        self.exchanger = None
        self.training_info = None
        self.initialized = False
        if ((input_dim is not None) and 
            (output_dim is not None)):
            # Initialize the input component positions.
            self.initialize(input_dim, location_dim, representation_dim, output_dim)
    
    def initialize(self, input_dim: int, output_dim: int,
                   representation_dim: Optional[int] = None,
                   location_dim: Optional[int] = None ):
        # Set a default for the representation and location dimensions.
        if (location_dim is None):
            location_dim = 4
        if (representation_dim is None):
            representation_dim = 2 * (input_dim + location_dim)
        # Store the integer size specifications.
        self.input_dim = input_dim
        self.location_dim = location_dim
        self.representation_dim = representation_dim
        self.output_dim = output_dim
        # Create the internal positions and the internal networks.
        self.positions = torch.zeros(input_dim, location_dim)
        self.positions = torch.as_tensor(random_ball(input_dim, location_dim, inside=False))
        self.aggregator = MLP(1+location_dim, representation_dim)
        self.exchanger = MLP(representation_dim, output_dim)
        # Record that this model has been initialized.
        self.initialized = True

    # Wrapper for the 'fit' function over this model.
    def fit(self, *args, **kwargs):
        return fit(self, *args, **kwargs)

    # Pass data through this network.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate the "position of the inputs" to the data.
        positions = torch.ones((x.shape[0], self.input_dim, self.location_dim))
        positions[:,:,:] *= self.positions # The last two dimensions match.
        # Flatten the positions array so that each position aligns with 1 component of an input.
        positions = torch.flatten(positions, 0, 1)
        # Flatten the input array so that each component is on its own row.
        px = torch.flatten(x, 0, 1).reshape((-1,1))
        # Concatenate the values from x with the positions.
        positions = torch.cat((px, positions), dim=1)
        aggregate = self.aggregator.forward(positions)
        # Reshape the aggregate to align with the original input data.
        aggregate = aggregate.reshape((x.shape[0],x.shape[1],self.representation_dim))
        # Take the mean "embedding" across all input components for each point.
        aggregate = torch.mean(aggregate, dim=1)
        # Pass the "embedded inputs" into the exchanger.
        output = self.exchanger.forward(aggregate)
        # Return the predicted output.
        return output

    # Save this model to a path.
    def save(self, path: str) -> None:
        torch.save(self, path)

    # Normalize inputs, then pass them forward through the network.
    #   Disable gradients for speed. Return Numpy array as output.
    def predict(self, x: List[List[float]]) -> List[List[float]]:
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x_normalized = (x - self.input_shift) / self.input_scale
            y_normalized = self.forward(x_normalized)
            y = y_normalized * self.output_scale + self.output_shift
        return y.detach().numpy()

    # When called with "model(<input>)" assume unnormalized data.
    def __call__(self, *args, **kwargs) -> List[List[float]]:
        return self.predict(*args, **kwargs)

# Construct a test function (with orthogonal output components when 2pi).
test_func = lambda x, d=10: np.asarray([
    np.cos(i * 2 * np.pi * np.linalg.norm(x, axis=1))
    for i in range(1,d+1)]).T

N = 1000
D_IN = 2
D_OUT = 3

# Test the regular network (no torch.jit.script for training function)
x = random_ball(N, D_IN)
y = test_func(x, D_OUT)
all_x, x, x_valid = x, x[:-N//10], x[-N//10:]
all_y, y, y_valid = y, y[:-N//10], y[-N//10:]
# m = MLP()
m = AggExc()
start_seconds = time.time()
m.fit(x, y, x_valid, y_valid)
end_seconds = time.time()
total_seconds = end_seconds - start_seconds
print("Time spent training:")
print(f"  {total_seconds:.2f} seconds")


from util.plot import Plot

# Plot the loss function and step sizes.
p = Plot()
step_range = list(range(1, m.training_info['optimizer_steps']+1))
p.add("Training MSE", step_range, m.training_info['training_loss'], mode="lines")
p.add("Validation MSE", step_range, m.training_info['validation_loss'], mode="lines")
p.add("Step sizes", step_range, m.training_info['step_sizes'], mode="lines")
_ = p.show(file_name="agg_exc_training.html")

# Plot the model outputs for each (orthogonal) output component.
for component in range(y.shape[1]):
    p = Plot()
    p.add("Data", *all_x.T, all_y[:,component])
    x_min_max = np.asarray([np.min(all_x, axis=0), np.max(all_x, axis=0)]).T
    p.add_func("Model", lambda x: m(x)[:,component], *x_min_max, vectorized=True)
    _ = p.show(file_name=f"agg_exc_out-{component+1}.html")
