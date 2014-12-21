.. 03_biopsyml:

==================================================
Machine learning with the ‘biopsy’ dataset, Part 2
==================================================

This is a continuation from the `Biopsy Machine Learning, Part 1 <http://sawcordwell.wordpress.com/2013/01/31/biopsy-machine-learning/>`_ post.
In this post I describe how I applied the my `backpropagation <http://sawcordwell.wordpress.com/2013/02/01/backpropagation-using-octave/>`_ functions to classify whether a breast cancer tumour is benign or malignant based on nine biopsy attributes.

I am using the ‘biopsy’ dataset from the MASS (Modern Applied Statistics with S) R Project package to use as the input and targets to train an artificial neural network using backpropagation.
Last time I mentioned that I thought the 'biopsy' dataset was nice to use for this purpose, but my explanation was probably a bit unclear.
The reason is that this data appeared to be easy to represent as the inputs and targets for the training algorithm without too much preprocessing.
The attributes have already all been abstracted in the same way and they also all are on the same scale.

Last time I left my training script at a bare ``for`` loop that didn't do anything except convert the data from ``biopsy`` to the inputs and target.
Now my goal is to finish that script using the backpropagation functions I had developed earlier.

After setting the variable ``biopsy`` to be global, we read in the biopsy dataset just like last time.
Then the activation vectors and weight matrices need to be created.
This can be done just like this from the Octave prompt (which later will be included in the script):

.. code-block:: octave

    [a, w] = makenet([90, 180, 180, 2]) ;

This means that the neural network will have 90 input neurons, 180 neurons each in two hidden layers and two neurons as output.
This seemed to be a good enough structure to me considering the ‘guidelines’ and ‘rules-of-thumb’ I have heard regarding network topology.
These guidelines seem to suggest using about one to three hidden layers, and two to three times as many neurons per hidden layer as there are input neurons.

I want to supplement my explanation from the first post as to why I chose 90 input neurons.
I wanted the input neurons to be binary so that at any one time all of them had values that were very similar.
When they are binary then the maximum difference between any two neurons is 1.
Again, this is due to the guidelines that I have heard about that all the data that is going into the network should be about the same scale.
I also would like to compare the performances of the cases when the input neurons take on their original values and when they are scaled from 0 to 1.
Therefore, these two cases would require only nine input neruons.
But I will leave that for another time when I post part three.

Back to the neural network. ``a`` is a cell matrix that contains one layer of neural activations per cell, and while ``w`` is also a cell matrix it contains one matrix of neural connection weights per cell.
``w`` could be used as it is with the ``feedforward`` function to calculate an output, but since the network has not been trained, it will not give nice results.
Assuming the working directory is the one with the backpropagation functions in it:

.. code-block:: octave

    rand('seed', 0)
    global biopsy
    biopsy = csvread("biopsy.csv", 1, 0) ;
    [a, w] = makenet([90, 180, 180, 2]) ;
    invec = getinput(1) ;
    target = gettarget(1) ;
    output = feedforward(invec, w) ;

Using the random seed of 0 will make this example repeatable.
The class from the first row of the biopsy data frame says that the tumour is benign, and the variable ``target`` shows this with vector ``[1, 0]``.
A one in the first element means benign, and a one in the seond element means malignant.
However, the variable ``output`` doesn't reflect this at all.
The vector is ``[0.72254, 0.70011]`` which doesn't even give an indication of which class is more likely.
So, the network needs to be trained.

There are two aspects of training a neural network, which are training and validating.
Validating is an important step to ensure that a neural network is general enough that it can apply to data that comes from another source.
I did this by splitting the data into two sections.
Roughly 75% of the data went into training and the remainder went into validating.
The training set is constructed by taking a random sample of 525 unique integers between 1 and 699.
These are then passed to the ``getinput`` function one-by-one as needed. The ``sample(n, m)`` function will return n samples between 0 and *m* − 1, so the addition of 1 is neccesary to correct this.
The validation set is all the other integers between 1 and 699 that are not in the training set, and the function ``setdiff`` is used to achieve this.
Another way of constructing this could have been to ensure that 75% of the benign examples and 75% of the malignant examples were included in the training set, however I chose to assume that they would be fairly evenly distributed.

.. code-block:: octave

    trainset = sample(525, 699) + 1 ;
    validset = setdiff(1:699, trainset) ;

Now we need to construct a training loop.
It's best not to loop through the training set just once and update the weights once for each training example.
There may need to be to be many iterations of training and validating before the neural network has been trained to a reasonable degree.
I usually construct iteration loops something along the lines of:

.. code-block:: octave

    done = false ;
    itr = 0 ;
    while ! done
        itr += 1 ;
        do_training()
        do_validating()
        if stopping_conditions_are_met
            done = true ;
        endif
    endwhile

We do the training in a ``for`` loop over the training set.
I decided that before each training loop I would shuffle up the order of the training set so that the weights aren't always adjusted in the same order.
I really don't know whether this might improve, degrade, or do nothing for performance.

.. code-block:: octave

    trainset = trainset(randperm(525)) ;
    for n=trainset
        invec = getinput(n) ;
        target = gettarget(n) ;
        a = forwardprop(invec, a, w) ;
        w = backprop(target, a, w) ;
    endfor

The validation loop is very similar, except that I also keep a track of how many validation examples had an error greater than some small number ``epsilon``.
Here I set ``epsilon`` to 0.01, which means that I am aiming for the squared error between target and output to be less than 0.01:

.. code-block:: octave

    nerrors = 0 ;
    for n=validset
        invec = getinput(n) ;
        target = gettarget(n) ;
        output = feedforward(invec, w) ;
        if max((target - output).^2) >= epsilon
            nerrors += 1 ;
        endif
    endfor

Finally, we checked whether conditions have been met to exit the ``while`` loop.
This code assumes that a ``maxiter`` variable has been defined previously:

.. code-block:: octave

    if nerrors == 0
        trained = true ;
        disp("Training completed, no errors during validation.")
    elseif itr == maxiter
        trained = true ;
        disp(["Stopping due to maxiter. " num2str(maxiter)  " iterations already \
            done."])
    endif

Basically this is all that is needed to train the neural network with the biopsy data.
The complete script 'biopsynn.m' is listed below, with a few additions to allow for plotting of error rates, and stopping when no progress is being made.
In part three, I will compare how three different ways of defining the inputs performs and perhaps look at the performance of different topologioes.

.. code-block:: octave

    #!/usr/bin/env octave

    clear all

    global biopsy ;
    biopsy = csvread("biopsy.csv", 1, 0) ;

    # Initialize the activation vectors and weight matrices to represent a
    # neural network.
    [a, w] = makenet([90, 180, 180, 2]) ;

    maxiter = 1000 ;
    itr = 0 ;
    trained = false ;
    epsilon = 0.01 ;

    # The frequency with which the mean error should be checked that it is 
    # unchanging
    checkf = 50 ;
    mnerror = -1 ;
    # The largest magnitude of error encountered during validation on each
    # iteration.
    maxerrors = zeros(maxiter, 1) ;
    # The number of errors greater than epsilon encountered during validation
    # on each iteration.
    nerrors = zeros(maxiter, 1) ;

    # The training set is 525 unique random samples from 1 to 699.
    trainset = (sample(525, 699) + 1) ;
    # The validation set is all the integers from 1 to 699 not in the training
    # set.
    validset = setdiff(1:699, trainset) ;

    while ! trained

        itr += 1 ; 

        # Training
        # I'm using randperm to shuffle the order of the set (so that it is not in 
        # ascending order), but I don't know if this is any differernt (ie better)
        # than just using the ascending order vector.
        trainset = trainset(randperm(525)) ;

        for n=trainset
            invec = getinput(n) ;
            target = gettarget(n) ;
            a = forwardprop(invec, a, w) ;
            w = backprop(target, a, w) ;
        endfor

        # Validation
        for n=validset
            invec = getinput(n) ;
            target = gettarget(n) ; 
            output = feedforward(invec, w) ;
            newerror = max((target - output).^2) ;
            maxerrors(itr) = max([newerror, maxerrors(itr)]) ;
            if newerror >= epsilon
                nerrors(itr) += 1 ;
            endif
        endfor 

        # The mean of the errors is checked every checkf iterations. If
        # mod(itr, checkf) is zero, then it is time to check it.
        checkitr = ! mod(itr, checkf) ;
        if checkitr
            mnerror = mean(nerrors((itr-checkf+1):itr)) ;
        endif

        # We can finish if some conditions are satisifid
        if nerrors(itr) == 0
            trained = true ;
            disp("Training completed, no errors during validation.")
        elseif checkitr && (nerrors(itr) == mnerror)
            trained = true ;
            disp(["Training completed, the error rate has stabilized to " \
            num2str(mnerror)  " errors per validation."])
        elseif itr == maxiter
            trained = true ;
            disp(["Stopping due to maxiter. " num2str(maxiter)  " iterations already \
                done."])
        endif

    endwhile

    figure()
    plot(maxerrors(1:itr))
    figure()
    plot(nerrors(1:itr))

    save('-binary', 'weights.oct', 'w')

