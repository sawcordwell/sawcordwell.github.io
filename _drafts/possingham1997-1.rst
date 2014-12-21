.. 04_possingham1997:

==================================================================================
Optimal fire management of a threatened species: Python MDP Toolbox worked example
==================================================================================
The [Possingham-and-Tuck-1997]_ was among the first to `apply Markov decision theory to a conservation biology problem <http://www.mssanz.org.au/MODSIM97/Vol 2/Possingham.pdf>`_.
Here I will follow their paper and replicate their results as a worked example of how to use PyMDPtoolbox, a `Markov decison process (MDP) toolbox <http://www.inra.fr/mia/T/MDPtoolbox/>`_ for Python.
For an introduction to MDPs see `Marescot et al (2013) <http://dx.doi.org/10.1111/2041-210X.12082>`_.

The first step is to download the toolbox from the `project repository <http://code.google.com/p/pymdptoolbox/>`_, or install from `PyPI <http://pypi.python.org>`_ with the ``pip install pymdptoolbox`` command.
If you downloaded the zip archive then unzip it and enter the `distutils <http://docs.python.org/2/library/distutils.html>`_ command ``python setup.py install`` into a console in pymdptoolbox's top-level directory.
Now you should be able to ``import mdptoolbox as mdp`` from the Python console.

Now, we need to get the paper and read through Section 1 and Section 2.
Okay, now that we know what the problem is we'll start by defining the state transition probabilities.
In this first post I will only concetrate on setting up the problem without spatial structure.
In a follow-up post, I will extend it to include the extra complexity of spatial structure.

The simplest aspect of the problem defined in the paper was how habitat suitability relates to the number of years since the last fire in a patch.
Figure 2 (of the paper) shows the relationship, and the equivalent Python code is as follows:

.. code-block:: python

    def getHabitatSuitability(years):
        """The habitat suitability of a patch relatve to the time since last fire.

        The habitat quality is low immediately after a fire, rises rapidly until
        five years after a fire, and declines once the habitat is mature. See
        Figure 2 in Possingham and Tuck (1997) for more details.

        Parameters
        ----------
        years : int
            Years since last fire.

        Returns
        -------
        r : float
            The habitat suitability.

        """
        assert years >= 0, "'years' must be a positive number"
        if years <= 5:
            return(0.2 * years)
        elif 5 <= years <= 10:
            return(-0.1 * years + 1.5)
        else:
            return(0.5)

The next part of the problem is how to define the states.
While there are a number of possible alternatives to storing the transition probabilities (such as a Python ``dict`` or a custom class), the way that PyMDPtoolbox uses them is through `NumPy <http://www.numpy.org>`_ arrays.
The transition probability array is given the name ``P`` and must be specified in a certain way.
It should have 3 dimensions, where the first dimension corresponds to the actions and the second dimension corresponds to the initial states before any transition, and the thrid dimension corresponds to the next states after any transition.
Each element of the array stores a probability: the probability that the state of the system transitions from the initial state :math:`s` to the next state :math:`s'` given that action :math:`a` was taken.
Assume that the total number of actions is :math:`A` and the total number of states is :math:`S`, then ``P`` is going to be an array with size :math:`A × S × S`.
Each slice of the array along the first dimenion is an :math:`S × S` matrix, which is the transition probability matrix for the corresponding action.

The indices of the rows of a transition probability matrix are the initial states, and the elements along the rows contain the probabilities that the system will transition to the state given by the index of the columns.
So, as we can see each state is uniquely identified by an index into the transition probability matrix.
Therefore we need a way to translate from the human readable state specified as a set of parameters, to the machine readable form of an index and vice-versa.
These two functions and the following globally defined constants will do the trick:

.. code-block:: python

    # The number of population abundance classes
    POPULATION_CLASSES = 7
    # The number of classes of years since a fire
    FIRE_CLASSES = 13
    # The number of states
    STATES = POPULATION_CLASSES * FIRE_CLASSES
    # The number of actions
    ACTIONS = 2

.. code-block:: python

    def convertStateToIndex(population, fire):
        """Convert state parameters to transition probability matrix index.

        Parameters
        ----------
        population : int
            The population abundance class of the threatened species.
        fire : int
            The time in years since last fire.

        Returns
        -------
        index : int
            The index into the transition probability matrix that corresponds to
            the state parameters.

        """
        assert 0 <= population < POPULATION_CLASSES, "'population' must be in " \
            "(0, 1...%d)" % POPULATION_CLASSES - 1
        assert 0 <= fire < FIRE_CLASSES, "'fire' must be in " \
            "(0, 1...%d) " % FIRE_CLASSES - 1
        return(population * FIRE_CLASSES + fire)

.. code-block:: python

    def convertIndexToState(index):
        """Convert transition probability matrix index to state parameters.

        Parameters
        ----------
        index : int
            The index into the transition probability matrix that corresponds to
            the state parameters.

        Returns
        -------
        population, fire : tuple of int
            ``population``, the population abundance class of the threatened
            species. ``fire``, the time in years since last fire.

        """
        assert index < STATES
        population = index // FIRE_CLASSES
        fire = index % FIRE_CLASSES
        return(population, fire)

I defined the constants globally at the top of the Python script.
[Possingham-and-Tuck-1997]_ specifically state that there are seven population abundance classes (``POPULATION_CLASSES``), whether class 0 corresponds to extinct.
They don't specifically mention how many years since fire classes there are, but judging by Figure 3 there are 13 (``FIRE_CLASSES``). A state is made up of a population class component and a years since fire class component, so to get the number of states (``STATES``) the number of both classes are multiplied together.
In Possingham and Tuck there are four actions, but since I am only considering a single population for now that means I only have two actions (``ACTIONS``).
Action ``0`` is do nothing, and action ``1`` is burn the forest patch.

Now we need a function that can return a row of the transition probability matrix defining all transition probabilities for a given state and action.
But before defining the function in full, let's break it up into the component pieces.
Note, any given transition can have a probability of zero, but each row of the transition probability matrix must sum to one; so with probability 1 any given state will transition to the next state (it can transition back to the same state).
The dynamics of the transition probabilities are given in Section 2.1 and Figure 1 of [Possingham-and-Tuck-1997]_.
We will need to set up a numpy vector array to store the probabilities, and also get the habitat suitability based on the years since last fire *F*::

    ...
    prob = np.zeros((STATES,))
    r = getHabitatSuitability(F)
    ...

The case years since last fire component is simply described: given action do nothing, *a* = 0, the years since last fire *F* will increase by one until it has reached the largest class, after which it is absorbed into the largest class; and given action burn, *a* = 1, the years since last fire *F* will go back to zero::

    ...
    if a == 0:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == 1:
        # When the patch is burned set the years since fire to 0.
        F = 0
    ...

Next is how the the abundance class will be affected.
This is broken into three components, first when the abundance class is zero (extinct), *x* = 0, second is when it is at a maximum, *x* = 6, and third is when it is between these two extremes.
If the population is extinct, then it stays extinct, so::

    ...
    new_state = convertStateToIndex(0, F)
    prob[new_state] = 1
    ...

If the abundance class is at the maximum, then it can either stay in the same class, with probability 1 − (1 − *s*)(1 − *r*), or move down a class, with probability (1 − *s*)(1 − *r*).
Also, if the burn action is performed, then the population abundance will move down one class::

    ...
    x_1 = x
    x_2 = x - 1
    if a == 1:
        x_1 -= 1
        x_2 -= 1
    new_state = convertStateToIndex(x_1, F)
    prob[new_state] = 1 - (1 - s) * (1 - r) # abundance stays the same
    new_state = convertStateToIndex(x_2, F)
    prob[new_state] = (1 - s) * (1 - r) # abundance goes down
    ...

When the population abundance is at an intermediate class, then the population can also move up a level.
If the abundance is at class 1 before the transition, then moving down a class will make it extinct.
If there is also a fire, then we need to make sure that the abundance class isn't set to −1 (an undefined value), so there is a check to only decrement the class if it is greater than zero.
In this case also, then ``x_2`` and ``x_3`` are equal, so the probabilities of these two states need to be summed::

    ...
    x_1 = x
    x_2 = x + 1
    x_3 = x - 1
    if a == 1:
        x_1 -= 1
        x_2 -= 1
        if x_3 > 0:
            x_3 -= 1
    new_state = convertStateToIndex(x_1, F)
    prob[new_state] = s # abundance stays the same
    new_state = convertStateToIndex(x_2, F)
    prob[new_state] = (1 - s) * r # abundance goes up
    new_state = convertStateToIndex(x_3, F)
    prob[new_state] += (1 - s) * (1 - r) # abundance goes down
    ...

And now here is the function in full:

.. code-block:: python

    def getTransitionProbabilities(s, x, F, a):
        """Calculate the transition probabilities for the given state and action.

        Parameters
        ----------
        s : float
            The probability of a population remaining in its current abundance
            class
        x : int
            The population abundance class
        F : int
            The number of years since a fire
        a : int
            The action to be performed

        Returns
        -------
        prob : array
            The transition probabilities as a vector from state (x, F) to every
            other state given action ``a`` is performed.

        """
        assert 0 <= x < POPULATION_CLASSES
        assert 0 <= F < FIRE_CLASSES
        assert 0 <= s <= 1
        assert 0 <= a < ACTIONS 
        prob = np.zeros((STATES,))
        r = getHabitatSuitability(F)
        # Efect of action on time in years since fire.
        if a == 0:
            # Increase the time since the patch has been burned by one year.
            # The years since fire in patch is absorbed into the last class
            if F < FIRE_CLASSES - 1:
                F += 1
        elif a == 1:
            # When the patch is burned set the years since fire to 0.
            F = 0
        # Population transitions
        if x == 0:
            # Demographic model probabilities
            # population abundance class stays at 0 (extinct)
            new_state = convertStateToIndex(0, F)
            prob[new_state] = 1
        elif x == POPULATION_CLASSES - 1:
            # Population abundance class either stays at maximum or transitions
            # down
            x_1 = x
            x_2 = x - 1
            # Effect of action on the state
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == 1:
                x_1 -= 1
                x_2 -= 1
            # Demographic model probabilities
            new_state = convertStateToIndex(x_1, F)
            prob[new_state] = 1 - (1 - s) * (1 - r) # abundance stays the same
            new_state = convertStateToIndex(x_2, F)
            prob[new_state] = (1 - s) * (1 - r) # abundance goes down
        else:
            # Population abundance class can stay the same, transition up, or
            # transition down.
            x_1 = x
            x_2 = x + 1
            x_3 = x - 1
            # Effect of action on the state
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == 1:
                x_1 -= 1
                x_2 -= 1
                # Ensure that the abundance class doesn't go to -1
                if x_3 > 0:
                    x_3 -= 1
            # Demographic model probabilities
            new_state = convertStateToIndex(x_1, F)
            prob[new_state] = s # abundance stays the same
            new_state = convertStateToIndex(x_2, F)
            prob[new_state] = (1 - s) * r # abundance goes up
            new_state = convertStateToIndex(x_3, F)
            # In the case when x_3 = 0 before the effect of an action is applied,
            # then the final state is going to be the same as that for x_1, so we
            # need to add the probabilities together.
            prob[new_state] += (1 - s) * (1 - r) # abundance goes down
        return(prob)

Now we loop over the states and actions, getting the transition probabilities and fill in the transition probability matrix.
The other important part of an MDP that we haven't discussed yet is the rewards.
The rewards depend on the state of the system, and are defind in [Possingham-and-Tuck-1997]_ to be zero if the population is extinct and one if the population is extant.
Therefore the rewards can be defined as a vector of length *S* (which is one of the valid ways of specifying rewards to PyMDPtoolbox).

.. code-block:: python

    def getTransitionAndRewardArrays(s):
        """Generate the fire management transition and reward matrices.

        The output arrays from this function are valid input to the mdptoolbox.mdp
        classes.

        Let ``S`` = number of states, and ``A`` = number of actions.

        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.

        Returns
        -------
        out : tuple
            ``out[0]`` contains the transition probability matrices P and
            ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
            numpy array and R is a numpy vector of length ``S``.

        """
        assert 0 <= s <= 1, "'s' must be between 0 and 1"
        # The transition probability array
        P = np.zeros((ACTIONS, STATES, STATES))
        # The reward vector
        R = np.zeros(STATES)
        # Loop over all states
        for idx in range(STATES):
            # Get the state index as inputs to our functions
            x, F = convertIndexToState(idx)
            # The reward for being in this state is 1 if the population is extant
            if x != 0:
                R[idx] = 1
            # Loop over all actions
            for a in range(ACTIONS):
                # Assign the transition probabilities for this state, action pair
                P[a][idx] = getTransitionProbabilities(s, x, F, a)
        return(P, R)

The hardest part is over and now we want to solve the MDP.
Fortunately with PyMDPtoolbox, solving an MDP is very straightforward.
The minimum amount of work that you have to do is choose which algorithm you want to use, and decide on a discount factor.
In this case Possingham and Tuck mention in Section 3 that they are using a back-stepping method with a timeframe of 50 years, so the appropriate PyMDPtoolbox class to use would be ``mdptoolbox.mdp.FiniteHorizon`` with the number of periods equal to 50.
The second item to decide upon is the discount factor.
The discount factor is the discount applied to future rewards.
It's not clear from the paper if [Possingham-and-Tuck-1997]_ used a discount factor or not, but I will set it to 0.96 which means that future rewards have a discount rate of 4%.

.. code-block:: python

    def solveMDP():
        """Solve the problem as a finite horizon Markov decision process.

        The optimal policy at each stage is found using backwards induction.
        Possingham and Tuck report strategies for a 50 year time horizon, so the
        number of stages for the finite horizon algorithm is set to 50. There is no
        discount factor reported, so we set it to 0.96 rather arbitrarily.

        Returns
        -------
        mdp : mdptoolbox.mdp.FiniteHorizon
            The PyMDPtoolbox object that represents a finite horizon MDP. The
            optimal policy for each stage is accessed with mdp.policy, which is a
            numpy array with 50 columns (one for each stage).

        """
        P, R = getTransitionAndRewardArrays(0.5)
        sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
        sdp.run()
        return(sdp)

Finally, we can run solve the MDP and have a look at the optimal policy.
The policy created by the finite horizon algorithm is an optimal policy for each period under consideration, it has the states as rows and each period as columns.
Possingham and Tuck consider policies 50 years from the terminal time, which represents a good long-term strategy.
To get this policy we take a slice from the policy array at the first column.

.. code-block:: python

    sdp = solveMDP()
    print(sdp.policy)
    print(sdp.policy[:, 0])

I have added a printing function that prints the policy as a table, which is replicated in the table below.
It shows for each population abundance, as the rows, and years since fire, as the columns, which action should be chosen.
The forest patch should not be burned until the population is in the highest abundance class and the time since last fire is seven years.
This strategy would result in a cycle of burning the forest at (6, 7) which will cause the state to move to (5, 0), the do no action as the state transitions through (6, 1), (6, 2)...(6, 7) and then burn again thereby restarting the cycle.

Output table:

=== === === === === === === === === === === === === ===
 \   Years since fire
--- ---------------------------------------------------
 \   0   1   2   3   4   5   6   7   8   9   10  11  12
=== === === === === === === === === === === === === ===
 0   0   0   0   0   0   0   0   0   0   0   0   0   0
 1   0   0   0   0   0   0   0   0   0   0   0   0   0
 2   0   0   0   0   0   0   0   0   0   0   0   0   0
 3   0   0   0   0   0   0   0   0   0   0   0   0   0
 4   0   0   0   0   0   0   0   0   0   0   0   0   0
 5   0   0   0   0   0   0   0   0   0   0   0   0   0
 6   0   0   0   0   0   0   0   1   1   1   1   1   1
=== === === === === === === === === === === === === ===

The `complete source code <https://bitbucket.org/scordwell/possingham1997>`_ is available on my blog’s `Bitbucket <http://www.bitbucket.org>`_ git repository.

.. [Possingham-and-Tuck-1997] Possingham H & Tuck G, 1997, ‘Application of stochastic
   dynamic programming to optimal fire management of a spatially structured
   threatened species’, *MODSIM 1997*, vol. 2, pp. 813–817. `Available online
   <http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf>`_.

