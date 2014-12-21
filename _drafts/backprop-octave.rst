.. 02_backprop_octave:

===============================
Backpropagation with GNU Ocatve
===============================

I recently spent some time writing my own functions to train neural networks using backpropagation.
I watched a few `Coursera Machine Learning Course <https://www.coursera.org/course/ml>`_ lectures by Andrew Ng and also the first few from `Caltech Machine Learning <http://www.caltech.edu/content/caltech-offers-online-course-live-lectures-machine-learning>`_ by Yaser Abu-Mostafa through iTunes U on my iPad.
Along with these I read or looked at `Devin McAuley's (UQ) <http://itee.uq.edu.au/~cogs2010/cmc/chapters/BackProp/index2.html>`_, `Daniel Crespin's (UCV) <http://www.matematica.ciens.ucv.ve/dcrespin/Pub/backprop.pdf>`_ and `Anand Venkataraman's <http://www.speech.sri.com/people/anand/771/html/node37.html>`_ documents and `Wikipedia <http://en.wikipedia.org/wiki/Backpropagation>`_.
Here I will outline how I have implemented the backpropagation algorithm in Octave.

The activation of the neurons in the network will be modelled with the sigmoid function.
The input ``f``, which can be direct input or the summed input from a number of other neurons, will be activated by the sigmoid function, which then becomes the output ``y`` of the neuron.
``f`` could be a matrix, vector or singleton.
In these examples I try to make use of vectorisation as much as possible, so ``f`` should be passed as a column vector, and ``y`` will be returned as a vector of the same dimensions.

.. code-block:: octave

    function y = activate(f)
    y = (1 ./ (1 + exp(-f))) ;
    endfunction

To be able to use the backpropagation algorithm, a representation of a neural network needs to be implemented.
Fortunately this is easy for simple artificial neural networks.
By simple I mean that all the neruons in one layer are connected to each neuron in the next layer and no neurons in any other layer.
Also, the activation functions for all neurons in all layers are the same.
This is also known as a fully connected multilayer perceptron.
My model has two parts to the neural network: first is the neural activations; and, second is the neural weights.

There is one activation vector for each layer of the network.
The first vector corresponds to the input layer, and takes on the values of the inputs.
The next vectors correspond to the hidden layers.
They take on values of the activated weighted sums of the previous vectors elements.
The final vector corresponds to the output layer, which is also an activated weighted sum of the previous layer.

As for the weights, there is one matrix of neural weights per layer of the network.
The weights correspond to the connections between a layer and the next layer.
The first matrix is the weights between the input and the first hidden layer.
The last matrix corresponds to the weights between the last hidden layer and the output layer.
If there are *L* layers, then there are going to be *L* − 1 matrices. Initially the weight matrices need to be created with small random values.
This is where the ``initweights`` function comes in: it accepts the number of rows (``r``) and columns (``c``) as input and returns a weight matrix that has values between −0.05 and 0.05.

.. code-block:: octave

    function weights = initweights(r, c)
    init = 0.05 ;
    weights = init - (2 * init) .* rand(r, c) ;
    endfunction

To set up the weights and activations we use the ``makenet`` function.
I decided that storing the weights and neuron activations in cell matrices was the best way.
That way it is easy to create a network with an arbitrary number of neurons in an arbitrary number of layers, where each cell corresponds to one layer.
The ``makenet`` function takes care of all this when it is passed a vector containing the number of neurons per layer.
The length of the vector ``sl`` is interpreted by the function to be the number of layers including the input, output and any hidden layers.
The elements of ``sl`` are interpreted to be the number of neurons per layer.
One more thing to note is that there are bias units in each activation layer, and the weights for these bias units are 1.

.. code-block:: octave

    function [activations, weights] = makenet(sl)
    L = length(sl) ;
    weights = cell(L, 1) ;
    activations = cell(L, 1) ;
    # Input layer
    activations{1} = [ 1 ; zeros(sl(1), 1) ] ;
    weights{1} = [ ones(sl(2), 1) initweights(sl(2), sl(1)) ] ;
    # Hidden layers
    for ii = 2:(L - 1)
        activations{ii} = [ 1 ; zeros(sl(ii), 1) ] ;
        weights{ii} = [ ones(sl(ii + 1), 1) initweights(sl(ii + 1), sl(ii)) ] ;
    endfor
    # Output layer
    activations(L) = zeros(sl(L), 1) ;
    # no weights for the output layer
    endfunction

To be able to use these as a feed-forward neural network, the function ``feedforward`` is defined to handle this.
``feedforward`` accepts an input vector and (pre-trained) neural weights as parameters.
It returns the output of the nerual network.
If ``weights`` have not previously been trained, then the output most likely will be meaningless.

.. code-block:: octave

    function output = feedforward(inputs, weights)
    inputs = inputs(:) ;
    L = length(weights) ;
    activations = cell(L, 1) ;
    activations{1} = [1 ; inputs] ;
    for ii = 2:(L - 1)
        activations{ii} = [1 ; activate(weights{ii - 1} * activations{ii - 1})] ;
    endfor
    output = activate(weights{L - 1} * activations{L - 1}) ;
    endfunction

As part of the training process the inputs of the training set need to be forward propagated through the network.
Since we are going to use backpropagation to adjust the neural weights, we will want to keep all the neural activations.
The function ``forwardprop`` takes as parameters the input vector, the activation vectors and the weight matrices.
It then returns all the activation vectors that result from the input.

.. code-block:: octave

    function activations = forwardprop(inputs, activations, weights)
    inputs = inputs(:) ;
    L = length(activations) ;
    activations{1}(2:length(activations{1})) = inputs ;
    for ii = 2:(L - 1)
        activations{ii}(2:length(activations{ii})) = activate(weights{ii - 1} * \
            activations{ii - 1}) ;
    endfor
    activations{L} = activate(weights{L - 1} * activations{L - 1}) ;
    endfunction

Once the activations are known, we backpropagate the error through the network and then update the weights accordingly.
The error is a measure of the difference between the actual output of the neural network compared to what was expected.
The back propagation function is called ``backprop`` and the parameters are ``target`` (the vector that the neural network should output), ``activations`` and ``weights``.

.. code-block:: octave

    function weights = backprop(target, activations, weights)
    learning_rate = 0.25 ;
    L = length(activations) ;
    deltas = calcdels(target, activations, weights) ;
    for ii = 1:(L - 1)
        s = length(activations{ii}) ;
        weights{ii}(:, 2:s) = weights{ii}(:, 2:s) + learning_rate .* \
            (deltas{ii + 1} * activations{ii}(2:s)') ;
    endfor
    endfunction

``backprop`` relies on the function ``calcdels`` to backpropagate the error and calculate how much each weight should be changed by.
``calcdels`` has the same parameters as ``backprop``.

.. code-block:: octave

    function deltas = calcdels(target, activations, weights)
    target = target(:) ;
    L = length(activations) ;
    deltas = cell(L, 1) ;
    deltas{L} = activations{L} .* (1 - activations{L}) .* \
        (target - activations{L}) ;
    for ii = (L - 1):-1:2
        s = length(activations{ii}) ;
        deltas{ii} = activations{ii}(2:s) .* (1 - activations{ii}(2:s)) .* \
            (deltas{ii + 1}' * weights{ii}(:, 2:s))' ;
    endfor
    endfunction

This is all the neccesary code required to write a script to train and validate a neural network.
In my next post I will continue to explain how I applied the neural network code shown here to classify/predict whether a breast cancer is beneign or malignant based on nine attributes of the tumour.

