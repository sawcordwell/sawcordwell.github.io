.. 01_biopsyml:

==================================================
Machine learning with the ‘biopsy’ dataset, Part 1
==================================================

I was looking through the datasets of the `R Project <http://www.r-project.org/>`_ CRAN package `Modern Applied Statistics with S (MASS) <http://cran.r-project.org/web/packages/MASS/index.html/>`_ to find one that I thought looked interesting to use in a machine learning context and I found ``biopsy``.
Potential benefits of using these data for testing machine learning algorithms include:

* The ouput (class of tumour) is binary.
* The inputs are all scaled from 1 to 10.
* There are quite a few samples (N=699).

Another dataset that looks interesting is ``Boston``, which I might look at another time.

My aim is to train a feed-forward neural network using backpropagation with the input attributes of *clump thickness*, *uniformity of cell size*, *uniformity of cell shape*, *marginal adhesion*, *single epithelial cell size*, *bare nuclei*, *bland chromatin*, *normal nucleoli* and *mitosis* to predict the output *class* being either ‘benign’ or ‘malignant’.
I probably should have looked at the references that are mentioned along with the documentation to get an idea of whether or not some attributes could be discarded because they don't significantly affect the class of tumour.
However, I decided to just jump straight in partly because I would like to try something with a lot of inputs (to see, for example, how well my code scales).

I don't have any of my own machine learning functions written for R, so I saved the dataset to a text file that I could use with the `GNU Octave <https://www.gnu.org/software/octave/>`_ functions that I had written a few weeks ago.
From the R command prompt I load the MASS library, then convert the biopsy class so that benign tumours get the value 0 and malignant tumours get the value 1.
This conversion is done so that Octave reads the data in properly, since matrices must have elements of all the same data type.
Then I wrote the data to file, excluding the first column which is a non-unique sample code number.

.. code-block:: r

    library(MASS)
    biopsy$class <- ifelse(biopsy$class == "benign", 0, 1))
    write.csv(biopsy[ , 2:11], "biopsy.csv", row.names=FALSE)

Then I closed R and fired up Octave.
In Octave the command to read in 'biopsy.csv' is: ``biopsy = csvread("biopsy.csv", 1, 0)``.
The second and third arguments to the ``csvread`` function mean that the csv file should be read from the second row and first column (they are zero-indexed). 
When ``write.csv`` is used with ``row.names=FALSE`` in R, then the column names are automatically written to the file and I haven't been able to turn them off.
By reading from the second row, we skip a row of incorrect zeros that otherwise would have been the first row.
By the way, there are sixteen values missing from the V6 (or bare nuclei) column of the data.
R saved them as ``NA`` but Octave reads them in as zero.
I decided to keep these rows for now, and leave the values as zero.

By now, we have a matrix ``biopsy`` that has 699 rows and 10 columns.
The next thing to decide is how to represent all this as a neural network.
I went with a neural network of 90 input neurons.
This is ten neurons for each attribute.
If the value of the attribute is one, then only the first neuron will take on a value of 1, and all the rest will be zero.
If the value is two, then the first two neurons take on the value of 1, and the rest zero.
And so on.
For the cases of the 16 missing values, no neurons will have the value one.
I am also thinking of trying just nine input neurons, where the value taken by the neuron will be scaled from 0 to one in steps of 0.1 corresponding to the value of the attribute from 0 to 10.
I may try this another time.

So, now I want to define a function that accepts a number between 1 and 699 that converts the information from a given row in biopsy to a column vector of length 90 representing the attributes for input to the neural network.
I called this function ``getinput`` and saved it to 'getinput.m'.

.. code-block:: octave

    function input = getinput(n)

    global biopsy ;
    input = zeros(90, 1) ;
    for k=1:9
        a = biopsy(n, k) ;
        b = (k * 10) - 9 ;
        input(b:(b + a - 1)) = ones(a, 1) ;
    endfor

    endfunction

And a similar function ``gettarget`` defined in 'gettarget.m':

.. code-block:: octave

    function target = gettarget(n)

    global biopsy ;
    target = zeros(2, 1) ;
    a = biopsy(n, 10) + 1 ;
    target(a) = 1 ;

    endfunction

I also created a file named 'biopsynn.m' whose contents so far are as follows.
I will call this script from Octave to run the neural network training.

.. code-block:: octave

    #!/usr/bin/env octave
    clear all ;

    global biopsy;
    biopsy = csvread("biopsy.csv", 1, 0) ;

    for n=1:699
        input = getinput(n) ;
        target = gettarget(n) ;
        # do training and validating here
    endfor
  
This is all for now.
In my next post I will describe my backpropagation functions, then try to apply them to the biopsy data.

