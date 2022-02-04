---
layout: post
title: Aggregate and Exchange Approximations
---

[//]: # (Formula can be generated at:
[//]: #   https://latex.codecogs.com/svg.image?latex_math_mode_code
[//]: # 
[//]: # Images can be included like this:
[//]: #   <img class="formula" src="./local-file.svg" title="name"/>
[//]: # 
[//]: # Visuals in the local director can be included like this:
[//]: #   <p class="visual">
[//]: #   <iframe src="./local-file.html">
[//]: #   </iframe>
[//]: #   </p>
[//]: #   <p class="caption">Caption under the visual.</p>
[//]: # 
[//]: # Everything else follows normal markdown syntax.


# Aggregate and Exchange Approximations
<p class="caption">Covering the 'sampled function' case with two multilayer perceptrons.</p>


The transformer has made inroads in many areas of applied approximation and its strong performance is often devoted to its "attention mechanism" that allows for pairwise comparison of input tokens. The work here aims to demonstrate that the truly powerful operation is not the specific combination of keys, queries, and values in an attention mechanism, but rather an effective placement of a `sum` operation that allows for order-independent processing.


Below is the code for a `torch` aggregate exchange architecture.


```python
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
    
    # Initialize the internal MLP's that form the aggregate and exchange operations.
    def initialize(self, input_dim: int, output_dim: int,
                   representation_dim: Optional[int] = None,
                   location_dim: Optional[int] = None ):
        # Set a default for the representation and location dimensions.
        if (location_dim is None):
            location_dim = 4  # <- research needs to be done here!
        if (representation_dim is None):
		  	   # As long as this is "large enough", the model should converge.
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
```


Here is the performance of the architecture when approximating a "pure" function, `cos(norm(x))`.


<p class="visual">
<iframe src="./agg_exc_training.html">
</iframe>
</p>
<p class="caption">The fit-time statistics for the aggregate exchange architecture.</p>


<p class="visual">
<iframe src="./agg_exc_out-1.html">
</iframe>
</p>
<p class="caption">The first component of the output prediction, `cos(2 pi norm(x))`.</p>


<p class="visual">
<iframe src="./agg_exc_out-2.html">
</iframe>
</p>
<p class="caption">The second component of the output prediction, `cos(norm(4 pi x))`.</p>


<p class="visual">
<iframe src="./agg_exc_out-3.html">
</iframe>
</p>
<p class="caption">The third component of the output prediction, `cos(norm(6 pi x))`.</p>
