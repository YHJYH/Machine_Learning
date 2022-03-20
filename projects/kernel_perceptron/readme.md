## Kernel perceptron (Handwritten Digit Classification) COMP0078 Supervised Learning Coursework
 
In this project, we train a classifier to recognize hand written digits, using the perceptron in two ways. First, we generalize the perceptron to use kernel function so that we may generate a nonlinear separating surface and second, we generalize the perceptron into a majority network of perceptrons so that instead of separating only two classes we may separate k classes.

The kernel we used is the polynomial kernel $K_{d}(\bold{p}.\bold{q}) = (\bold{p}\times\bold{q})^{d}$ which is parameterized by a positive integer $d$ controlling the dimension of the polynomial.

The algorithm is *online* that is the algorithms operate on a single example $(\bold{x}_{t},y_{t})$ at a time.

Data are from [here]( http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/). Relevant datasets are 
