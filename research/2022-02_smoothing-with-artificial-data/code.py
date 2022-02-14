import numpy as np
from tlux.plot import Plot
from tlux.random import well_spaced_box
from tlux.approximate.plrm import PLRM
from tlux.approximate.delaunay import Delaunay

# A function for testing approximation algorithms.
def f(x):
    x = x.reshape((-1,2))
    x, y = x[:,0:1], x[:,1:2]
    return 3*x + np.cos(8*x)/2 + np.sin(5*y)


if __name__ == "__main__":

    np.random.seed(0)

    print("Running code..")
    # Get random points that are well spaced in the unit [0,1] box in 2 dimensions.
    d = 2
    small_n = 15
    big_n = 300
    x = well_spaced_box(small_n, d)
    x_min_max = np.vstack((np.min(x,axis=0), np.max(x, axis=0))).T
    # Evaluate the test function at those points.
    y = f(x)


    # MLP model config.
    config = dict(
        di = 2, # input dimension
        do = 1, # output dimension
        ds = 16, # internal state dimension
        ns = 8, # num layers / internal states
        num_threads = 2,
        seed = 0,
        steps = 1000,
    )

    
    # ----------------------------------------------------------------
    #                        test-function.html
    # 
    p = Plot("")
    p.add_func("f", f, [0,1], [0,1])
    p.plot(file_name="test-function.html", show=True)


    # ----------------------------------------------------------------
    #                        fit-surface.html
    # 
    # Build a model over the data and show the fit.
    print()
    print("Fitting a model to a small amount of data..")
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    m = PLRM(**config)
    m.fit(x, y)
    p = Plot()
    p.add("n=16", *x.T, y[:,0])
    p.add_func("fit", m, *x_min_max)
    p.plot(file_name="fit-surface.html", show=False)
    # Add a visual that shows the fit error progression.
    p = Plot("Mean Squared Error and Step Sizes")
    p.add("MSE", list(range(config["steps"])), m.record[:,0], color=1, mode="lines")
    p.add("Step sizes", list(range(config["steps"])), m.record[:,1], color=2, mode="lines")
    p.plot(append=True, show=True)


    # ----------------------------------------------------------------
    #                    fit-surface-delaunay.html
    # 
    # Initialize a model for approximating the true function and extra function.
    print()
    print("Fitting model with additional data added in a new category..")
    config["ne"] = 2 # set the number of embeddings
    m = PLRM(**config)
    # Generate new data, use a piecewise linear fit of source data to approximate.
    x2 = well_spaced_box(big_n, d)
    piecewise_linear = Delaunay()
    piecewise_linear.fit(x, y)
    y2 = piecewise_linear(x2)
    print("x2.shape: ", x2.shape)
    print("y2.shape: ", y2.shape)
    # Concatenate all data together to train the new (piecewise linear biased) model.
    all_x = np.concatenate((x, x2), axis=0)
    all_y = np.concatenate((y, y2), axis=0)
    all_xi = np.concatenate(
        (np.ones((len(x),1), dtype="int32"),
         2*np.ones((len(x2),1), dtype="int32")),
        axis=0)
    print("all_x.shape:  ", all_x.shape)
    print("all_y.shape:  ", all_y.shape)
    print("all_xi.shape: ", all_xi.shape, " these are the integer encodings, either 1 or 2")
    m.fit(all_x, all_y, xi=all_xi)
    # Generate the visual of the new fit.
    print()
    print("Generating surface plot..")
    p = Plot()
    p.add("xi=1 (n=15)", *x.T, y[:,0], color=0)
    p.add_func("xi=1", lambda x: m(x, np.ones(len(x), dtype="int32").reshape((1,-1))), [0,1], [0,1])
    p.add("xi=2 (n=300)", *x2.T, y2[:,0], color=1, marker_size=4, marker_line_width=1, shade=True)
    p.plot(file_name="fit-surface-delaunay.html", show=False)
    # Add a visual that shows the fit error progression.
    p = Plot("Mean Squared Error and Step Sizes")
    p.add("MSE", list(range(config["steps"])), m.record[:,0], color=1, mode="lines")
    p.add("Step sizes", list(range(config["steps"])), m.record[:,1], color=2, mode="lines")
    p.plot(append=True, show=True)

