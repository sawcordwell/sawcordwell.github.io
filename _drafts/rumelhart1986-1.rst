.. 05_rumelhart1986:

============================================
Training a neural network to detect symmetry
============================================

[Rumelhart-etal-1986]_ described `the back-propagation algorithm <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_ in the journal `Nature` in 1986.
Since then the method has become an important part of machine learning.
Here I attempt to replicate the results they presented in Figure 1 of their paper using `GNU Octave <https://www.gnu.org/software/octave/>`_.
In a few previous posts from last year, I looked at using neural networks in GNU Octave.
However, I now consider that code to be quite inefficient and poorly written.
Furthermore it is actually incorrect in several places.

The problem that is being addressed is to create a function that can classify an input vector as either symmetric or not.
The first code that we shall write is generating the input data.
The input vector consists of six input units that can take on binary values.
Therefore there are :math:`2^6 = 64` possible input vectors. 
This data can be generated with these three lines of code:

.. code-block:: octave

    f = @(x) str2double(x) ;
    X = dec2bin(0:(2^6-1)) ;
    X = arrayfun(f, X) ;

`dec2bin <http://octave.sourceforge.net/octave/function/dec2bin.html>`_ is a function that converts an integer to its binary representation as a string.
The `anonymous function <http://www.gnu.org/software/octave/doc/interpreter/Anonymous-Functions.html>`_ ``f`` and `arrayfun <http://www.gnu.org/software/octave/doc/interpreter/Function-Application.html>`_ help to vectorize the code and convert each digit from a string to a number using `str2double <http://octave.sourceforge.net/octave/function/str2double.html>`_ and place it as an element of ``X``.
So now we have a 64 by 6 matrix where each row corresponds to a training example and each column corresponds to an input unit.
The goal is to train a neural network by back-propagation to classify each vector as either symmetric or not.
To do this training we must first classify each input manually to be able to teach the network.
This line of code can tell us whether they are symmetric or not:

.. code-block:: octave

    Y = all(X(:, 1:3) == fliplr(X(:, 4:6)), 2) ;

We want to detect whether the vectors are symmetric about the centre point.
The `fliplr <http://octave.sourceforge.net/octave/function/fliplr.html>`_ function is flipping the last 3 columns of each vector from left to right so that the equality operator is checking the first unit against the sixth, the second against the fifth and the third against the fourth.
The `all <http://octave.sourceforge.net/octave/function/all.html>`_ function, with the second argument of 2 is testing whether each row contains only true values.
So ``Y`` is a column vector where each element indicates whether the corresponding row vector in ``X`` is symmetric about the centre point.
There are only 8 symmetric vectors, and the rest are asymmetric.

[Rumelhart-etal-1986]_ used gradient descent to find optimal values for the network weights.
Gradient descent aims to minimise a cost over a set of parameters of a function by taking steps in the direction of the gradient.
The parameters are the set of neural network weights, and the cost is defined in terms of the errors on the output and the gradients are calculated using back-propagation.

The network has three layers, the input layer, one hidden layer and an output layer.
The first layer has six nodes for input plus a bias node, the second layer has two nodes plus a bias node, and the last layer has one output node.
The input vector :math:`x` is propagated forward through each layer of the network.
The input units, :math:`x_j`, to a layer are a linear function of the values of the units in the output vector, :math:`y_i`, of the previous layer and the weights, :math:`w_{ji}`, connecting the previous layer to the current layer.
The equation is:

.. math::

    x_j = \sum_i y_i w_{ji}

Or in matrix notation: :math:`x = w\cdot y`, where :math:`w` is a matrix and :math:`y` is a column vector.

Each unit has an output :math:`y_j` that is a non-linear function of its total inputs :math:`x_j`: :math:`y_j = \frac{1}{1 + e^{-x}}`, otherwise known as the `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_.

So now we'll write the code for generating the initial weight matrices, and also feeding and input vector through the network to get output.
The initial weights in the paper were distributed according to the uniform random distribution between −0.3 and 0.3.

.. code-block:: octave

    s = [6, 2, 1] ;
    w = cell(2, 1) ;
    w{1} = 0.6 * rand(s(2), s(1) + 1) - 0.3 ;
    w{2} = 0.6 * rand(s(3), s(2) + 1) - 0.3 ;

``s`` represents the neural networks layer architecture, so the first layer has 6 units, the second has 2 and the third has 1.
I am going to store the weights in a cell matrix called ``w``.
The first set of weights connects the input layer to the hidden layer.
The dimension of the weight matrix has to take account of the bias unit, so the dimension is the number of units in the hidden layer by the number of units in the input layer plus one.
In other words the size of ``w`` is ``[2, 6 + 1]`` = ``[2, 7]``.
These dimensions are given to `rand <http://octave.sourceforge.net/octave/function/rand.html>`_ to produce uniformly distributed matrix that is 2 by 7.
The second weight matrix must be 1 by 3 taking into account the bias unit of the hidden layer.

First I define the non-linear sigmoid activation function:

.. code-block:: octave

    function x = sigmoid(x)
    x = 1 ./ (1 + exp(-x)) ;
    endfunction

Then the code to calculate the activations of the units in each layer:

.. code-block:: octave

    m = size(X, 1) ;
    a = cell(3, 1) ;
    a{1} = [ones(m, 1) X] ;
    a{2} = [ones(m, 1) sigmoid(a{1} * (w{1})')] ;
    a{3} = sigmoid(a{2} * (w{2})') ;

The total error :math:`E` was defined as:

.. math::

    E = \frac{1}{2} \sum_c \sum_j (y_{j,c} - d_{j,c})^2

or in octave code: :code:`error = 0.5 * sumsq(a{3} - Y) ;`

The partial derivatives :math:`∂E/∂x`:

.. code-block:: octave

    delta = cell(3, 1) ;
    delta{3} = (a{3} - Y) .* a{3} .* (1 - a{3}) ;
    delta{2} = (delta{3} * w{2}(:, 2:end)) .* a{2}(:, 2:end) .* (1 - a{2}(:, 2:end)) ;

The gradients of the weights :math:`∂E/∂w`:

.. code-block:: octave

    grad = cell(2, 1) ;
    grad{1} = (delta{2})' * a{1} ;
    grad{2} = (delta{3})' * a{2} ;

Gradient descent was used in the paper.
It would be possible (and convergence would probably be faster) to use more advanced optimisation routines, such as Octaves built-in `fminunc <http://octave.sourceforge.net/octave/function/fminunc.html>`_.
However, I decided to stick to the original method as much as possible; so I wrote a simple gradient descent function.
The main thing to point out is that the stopping criteria differs from what is usually used in optimisation problems.
Instead of stopping when the change in cost is below some tolerance, I stop once the neural net's predictions match the expected output.
This decision was made to compare the number of sweeps required through all the cases using my method with the number that [Rumelhart-etal-1986]_ reported was required.
(Although I am not sure what their stopping condition was, so a direct comparison would not be possible.)

.. code-block:: octave

    function [weights cost ITR] = gradientDescent(weights, X, Y, max_iter)
    global s;
    epsilon = 0.1 ;
    alpha = 0.9 ;
    change = zeros(((s(1) + 1) * s(2)) + ((s(2) + 1) * s(3)), 1) ;
    for ITR = 1:max_iter
        [error grad] = computeTotalError(weights, X, Y) ;
        % ∆w(t) = -ε∂E/∂w(t) + α∆w(t-1)
        change = -epsilon * grad + alpha * change ;
        weights += change ;
        if all(predict(X, reshapeLayers(weights)) == Y)
            break ;
        endif
    endfor
    endfunction

To use this function, I will need to define three more functions ``computeTotalError``, ``reshapeLayers``, and ``predict``.
The simplest is ``predict``, which just does forward propagation and returns the output layer.

.. code-block:: octave

    function h = predict(X, weights)
    m = size(X, 1) ;
    h = sigmoid([ones(m, 1) X] * (weights{1})') ;
    h = sigmoid([ones(m, 1) h] * (weights{2})') ;
    h = round(h) ;
    endfunction

The ``predict`` function will return the hypothesis ``h``, which is the rounded output output of the neural network.
It assumes that ``X`` is a ``m`` by ``n`` matrix wherre ``m`` is the number of cases and ``n`` is the number of features.
So each row of ``X`` is a vector of input features to the network.
This corresponds to how ``X`` was defined above.
It also assumes that ``weights`` is a cell matrix with two elements where each cell contains a matrix that is appropriate for the networks layer structure.
The number of columns in ``weights{1}`` must be ``n + 1``.
That the hypothesis is rounded means that the ouput will be correct once it is on the correct side of 0.5.

The ``gradientDescent`` function accepts weights as a vector.
This means that to work with the feed-forward and back-propagation the weights will have to be reshaped and then unrolled:

.. code-block:: octave

    function cell_mat = reshapeLayers(v)
    global s;
    cell_mat = cell(2, 1) ;
    a = 1 ;
    for l = 1:2
        b = a - 1 + (s(l + 1) * (s(l) + 1)) ;
        cell_mat{l} = reshape(v(a:b), s(l + 1), s(l) + 1) ;
        a = b + 1 ;
    endfor
    endfunction

.. code-block:: octave

    function v = unrollLayers(cell_mat)
    v = [] ;
    for l = 1:size(cell_mat, 1)
        v = [v; cell_mat{l}(:)] ;
    endfor
    endfunction

Now it comes to the ``computeTotalError`` function, which will tie in much of the code from earlier:

.. code-block:: octave

    function [error grad] = computeTotalError(w, X, Y)
    % Number of training examples
    m = size(X, 1) ;
    w = reshapeLayers(w) ;
    % Forward propagation
    a = cell(3, 1) ;
    a{1} = [ones(m, 1) X] ;
    a{2} = [ones(m, 1) sigmoid(a{1} * (w{1})')] ;
    a{3} = sigmoid(a{2} * (w{2})') ;
    % The total error, E = ½ ∑ₘ∑ⱼ(aⱼₘ - yⱼₘ)²
    error = 0.5 * sumsq(a{3} - Y) ;
    % Backpropagate the errors, ∂E/∂x
    delta = cell(3, 1) ;
    delta{3} = (a{3} - Y) .* a{3} .* (1 - a{3}) ;
    delta{2} = (delta{3} * w{2}(:, 2:end)) .* a{2}(:, 2:end) .* (1 - a{2}(:, 2:end)) ;
    % The error gradients of the weights, ∂E/∂wₖⱼ = ∂E/∂xₖ · yⱼ
    grad = cell(2, 1) ;
    grad{1} = (delta{2})' * a{1} ;
    grad{2} = (delta{3})' * a{2} ;
    grad = unrollLayers(grad) ;
    endfunction

In their paper, they used

.. math::

    \Delta w(t) = -ε∂E/∂w(t) + α\Delta w(t - 1)

as a method to help faster convergence than

.. math::

    \Delta w = -ε∂E/∂w

I did the same and stored :math:`\Delta w(t-1)` to ``grad_prev``, which was declared a global variable.
The other global variables that I define are ``epsilon``, ``alpha`` and ``s`` so that these can be used throughout the functions as needed.

[Rumelhart-etal-1986]_ reported that it took 1425 sweeps through the entire dataset to learn.
I found that it was very variable and sometimes would learn after just a couple of hundred sweeps, while at other times it took tens of thousands sweeps.
I set the random number generator state using :code:`rand('state', 1)` and completed 1000 iterations of initialising random weights then using gradient descent until the output from ``predict`` agreed wih ``Y`` for the first time.
I recorded the number of iterations it took and the final cost.
Here are some summary statistics:

========== ======= ============== ======= ============== ======== ======= ================== ======== ========
    \      Minimum First Quartile Median  Third Quartile Maximum  Mean    Standard Deviation Skewness Kurtosis
---------- ------- -------------- ------- -------------- -------- ------- ------------------ -------- --------
Iterations  333     1127           2715    7829.75        54770   5891.91  7325.30            2.28426  6.45080
Error      0.44754  1.63054       1.67648  1.70834        1.86460 1.59441 0.24843            -2.32282  4.22037
========== ======= ============== ======= ============== ======== ======= ================== ======== ========

Finally, here is a diagram of the final weights for each layer of the network from a random sweep through the cases.

I quote from [Rumelhart-etal-1986]_ for the explanation:
    The key property of this solution is that for a given hidden unit, weights are symmetric about the middle of the input vector are equal in magnatude and opposite in sign.
    So if a symmetrical pattern is presented, both hidden units will receive a net input of 0 from the input units, and, because the hidden units have a negative bias, both will be off.
    In this case the output unit, having a positive bias, will be on.
    Note that the weights on each side of the midpoint are in the ratio 1:2:4.
    This ensures that each of the eight patterns that can occur above the midpoint sends a unique activation sum to each hidden unit, so the only pattern below the midpoint that can exactly balance this sum is the symmetrical one.
    For all non-symmetrical patterns, both hidden units will receive non-zero activations from the input units.
    The two hidden units have identical patterns of weights but with opposite signs, so for every non-symmetric pattern one hidden unit will come on and suppress the output.

The `complete source code <https://bitbucket.org/scordwell/rumelhart1986/src>`_ can be downloaded on Bitbucket.

.. [Rumelhart-etal-1986] Rumelhart DE, Hinton GE & Williams RJ, 1986,
   ‘Learning representations by back-propagating errors’, `Nature`,
   vol. 323, pp. 533–536. DOI: `10.1038/323533a0 <http://dx.doi.org/10.1038/323533a0>`_.

