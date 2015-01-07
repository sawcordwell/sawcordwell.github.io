---
layout: post
title: Optimal fire management of a threatened species, part 1
subtitle: Python MDP Toolbox worked example
date: 2015-01-10 15:00:00
categories: mdp conservation
tags: [MDP, Python, Markovian, decision theory, toolbox, tutorial]
---

The paper by [Possingham and Tuck (1997)][poss-tuck-97] was among the first to apply Markov decision theory to a conservation biology problem.
Here I will follow their paper and replicate their results as a worked example of how to use PyMDPtoolbox, a [Markov decison process (MDP) toolbox](<http://www.inra.fr/mia/T/MDPtoolbox/>) for Python.
For an introduction to MDPs see [Marescot _et al_ (2013)](<http://dx.doi.org/10.1111/2041-210X.12082>).

The complete source code for this tuorial is available in a [GitHub gist]().

## Setup

The first step is to install the PyMDPtoolbox package.
The repository for PyMDPtoolbox is hosted on GitHub at <https://github.com/sawcordwell/pymdptoolbox>, where you can find more detailed installation instructions.
Briefly you can install from [PyPI](<http://pypi.python.org>) with the `pip install pymdptoolbox` command or clone the repository with `git clone https://github.com/sawcordwell/pymdptoolbox.git`.
If you do not have [pip](https://pypi.python.org/pypi/pip) or [Git](http://git-scm.com/) installed, then I recommend that you install both.
If you still do not want to go to the trouble then download the zipped archive from [PyMDPtoolbox's GitHub page](https://github.com/sawcordwell/pymdptoolbox).
If you downloaded the zip archive then unzip it and enter the [setuptools](http://docs.python.org/2/library/distutils.html) command `python setup.py install` into a console in pymdptoolbox's top-level directory.
Now you should be able to `from mdptoolbox import mdp` in the Python console.

First, we need to download the [paper][poss-tuck-97] and read through Section&nbsp;1 and Section&nbsp;2.
Okay, now that we know what the problem is we'll start by defining the state transition probabilities.
In this first post I will only concetrate on setting up the problem without spatial structure.
This means that there will only be one population.
In a follow-up post, I will extend it to include the extra complexity of spatial structure with two populations.

The following modules are required and need to be imported:

{% highlight python %}
import numpy as np

from mdptoolbox import mdp
{% endhighlight %}

Define the following constants globally, they are used to specify the dimensions of the problem.
[Possingham and Tuck (1997)][poss-tuck-97] specifically state that there are seven population abundance classes (`POPULATION_CLASSES`), where class 0 corresponds to extinct.
They don't specifically mention how many classes there are to represent the years since last fire, but judging by Figure&nbsp;3 there are 13 (`FIRE_CLASSES`).
A state is made up of a population class component and a years since fire class component, so the number of states (`STATES`) is the number of both classes multiplied together.
In Possingham and Tuck there are four actions, but since I am only considering a single population for now that means I only have two actions (`ACTIONS`).
Action `0` is do nothing, and action `1` is burn the forest patch.

{% highlight python %}
# The number of population abundance classes
POPULATION_CLASSES = 7
# The number of classes of years since a fire
FIRE_CLASSES = 13
# The number of states
STATES = POPULATION_CLASSES * FIRE_CLASSES
# The number of actions
ACTIONS = 2
{% endhighlight %}

## Habitat suitability

The simplest aspect of the problem defined in the paper was how habitat suitability relates to the number of years since the last fire in a patch.
Figure&nbsp;2 (of the paper) shows the relationship, and the equivalent Python code is as follows:

{% highlight python %}
def get_habitat_suitability(years):
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
        return 0.2*years
    elif 5 <= years <= 10:
        return -0.1*years + 1.5
    else:
        return 0.5
{% endhighlight %}

## States, actions and rewards

The next part of the problem is how to define the states and rewards.
While there are a number of possible alternatives to storing transition probabilities (such as a Python `dict` or a custom class), the way that PyMDPtoolbox uses them is through [NumPy](<http://www.numpy.org>) arrays.
The transition probability array is given the name `P` and must be specified in a certain way.
It should have 3 dimensions, where the first dimension corresponds to the actions and the second dimension corresponds to the initial states before any transition, and the thrid dimension corresponds to the next states after any transition.
Each element of the array stores a probability: the probability that the state of the system transitions from the initial state <math>s</math> to the next state <math>s'</math> given that action <math>a</math> was taken.
Let the total number of actions be <math>A</math> and the total number of states be <math>S</math>, then ``P`` is going to be an array with size <math>A × S × S</math>.
Each slice of the array along the first dimenion is an <math>S × S</math> matrix, which is the transition probability matrix for the corresponding action.

The reward `R` can be either a vector of length <math>S</math> or a numpy array of size <math>S × A</math>.
... etc...

### Probability transition matrices

Let us look at a simple example before moving on...

{% highlight python %}
P = np.array([[[0.5, 0.5],
               [0.8, 0.2]],
              [[0.0, 1.0],
               [0.1, 0.9]]])
{% endhighlight %}

... blah blah...

The indexes of the rows of a transition probability matrix are the initial states, and the elements along the rows contain the probabilities that the system will transition to the state given by the index of the columns.
So, as we can see each state is uniquely identified by an index into the transition probability matrix.
Therefore we need a way to translate from the human readable state specified as a set of parameters, to the machine readable form of an index and vice-versa.
These two functions and the globally defined constants will do the trick:

{% highlight python %}
def convert_state_to_index(population, fire):
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
    return population * FIRE_CLASSES + fire
{% endhighlight %}

{% highlight python %}
def convert_index_to_state(index):
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
    assert 0 <= index < STATES
    population = index // FIRE_CLASSES
    fire = index % FIRE_CLASSES
    return (population, fire)
{% endhighlight %}

The case years since last fire component is simply described: given action do nothing, <math>a = 0</math>, the years since last fire <math>F</math> will increase by one until it has reached the largest class, after which it is absorbed into the largest class; and given action burn, <math>a = 1</math>, the years since last fire <math>F</math> will go back to zero:

{% highlight python %}
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
{% endhighlight %}

Now we need a function that can return a row of the transition probability matrix defining all transition probabilities for a given state and action.
But before defining the function in full, let's break it up into the component pieces.
Note, any given transition can have a probability of zero, but each row of the transition probability matrix must sum to one; so with probability 1 any given state will transition to the next state (it can transition back to the same state).
The dynamics of the transition probabilities are given in Section 2.1 and Figure 1 of [Possingham and Tuck (1997)][poss-tuck-97].
We will need to set up a numpy vector array to store the probabilities, and also get the habitat suitability based on the years since last fire <math>F</math>:

{% highlight python %}
...
prob = np.zeros(STATES)
r = getHabitatSuitability(F)
...
{% endhighlight %}

Next is how the the abundance class will be affected.
This is broken into three components, first when the abundance class is zero (extinct), <math>x = 0</math>, second is when it is at a maximum, <math>x = 6</math>, and third is when it is between these two extremes.
If the population is extinct, then it stays extinct, so:

{% highlight python %}
...
new_state = convertStateToIndex(0, F)
prob[new_state] = 1
...
{% endhighlight %}

If the abundance class is at the maximum, then it can either stay in the same class, with probability <math>1 − (1 − s)(1 − r)</math>, or move down a class, with probability <math>(1 − s)(1 − r)</math>.
Also, if the burn action is performed, then the population abundance will move down one class:

{% highlight python %}
...
# Population abundance class either stays at maximum or transitions
# down
transition_same = x
transition_down = x - 1
# If action 1 is taken, then the patch is burned so the population
# abundance moves down a class.
if a == 1:
    transition_same -= 1
    transition_down -= 1
# transition probability that abundance stays the same
new_state = convert_state_to_index(transition_same, F)
prob[new_state] = 1 - (1 - s) * (1 - r)
# transition probability that abundance goes down
new_state = convert_state_to_index(transition_down, F)
prob[new_state] = (1 - s) * (1 - r)
...
{% endhighlight %}

When the population abundance is at an intermediate class, then the population can also move up a level.
If the abundance is at class 1 before the transition, then moving down a class will make it extinct.
If there is also a fire, then we need to make sure that the abundance class isn't set to `−1` (an undefined value), so there is a check to only decrement the class if it is greater than zero.
In this case also, then `x_2` and `x_3` are equal, so the probabilities of these two states need to be summed:

{% highlight python %}
...
# Population abundance class can stay the same, transition up, or
# transition down.
transition_same = x
transition_up = x + 1
transition_down = x - 1
# If action 1 is taken, then the patch is burned so the population
# abundance moves down a class.
if a == 1:
    transition_same -= 1
    transition_up -= 1
    # Ensure that the abundance class doesn't go to -1
    if transition_down > 0:
        transition_down -= 1
# transition probability that abundance stays the same
new_state = convert_state_to_index(transition_same, F)
prob[new_state] = s
# transition probability that abundance goes up
new_state = convert_state_to_index(transition_up, F)
prob[new_state] = (1 - s) * r
# transition probability that abundance goes down
new_state = convert_state_to_index(transition_down, F)
# In the case when transition_down = 0 before the effect of an action
# is applied, then the final state is going to be the same as that for
# transition_same, so we need to add the probabilities together.
prob[new_state] += (1 - s) * (1 - r)
...
{% endhighlight %}

And now here is the function in full:

{% highlight python %}
def get_transition_probabilities(s, x, F, a):
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

    assert 0 <= x < POPULATION_CLASSES, \
        "x not in {0, 1, …, %d}, x = %s" % (POPULATION_CLASSES - 1, str(x))
    assert 0 <= F < FIRE_CLASSES, \
        "F not in {0, 1, …, %d}, f = %s" % (FIRE_CLASSES - 1, str(F))
    assert 0 <= s <= 1, "s not in [0, 1], s = %s" % str(s)
    assert 0 <= a < ACTIONS, \
        "a not in {0, 1, …, %d}, a = %s" % (ACTIONS - 1, str(a))

    # a vector to store the transition probabilities
    prob = np.zeros(STATES)
    # the habitat suitability value
    r = get_habitat_suitability(F)

    ## Efect of action on time in years since fire.
    if a == 0:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == 1:
        # When the patch is burned set the years since fire to 0.
        F = 0

    ## Population transitions
    if x == 0:
        # Demographic model probabilities
        # population abundance class stays at 0 (extinct)
        new_state = convert_state_to_index(0, F)
        prob[new_state] = 1
    elif x == POPULATION_CLASSES - 1:
        # Population abundance class either stays at maximum or transitions
        # down
        transition_same = x
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == 1:
            transition_same -= 1
            transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = 1 - (1 - s) * (1 - r)
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        prob[new_state] = (1 - s) * (1 - r)
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        transition_same = x
        transition_up = x + 1
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == 1:
            transition_same -= 1
            transition_up -= 1
            # Ensure that the abundance class doesn't go to -1
            if transition_down > 0:
                transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = s
        # transition probability that abundance goes up
        new_state = convert_state_to_index(transition_up, F)
        prob[new_state] = (1 - s) * r
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        # In the case when transition_down = 0 before the effect of an action
        # is applied, then the final state is going to be the same as that for
        # transition_same, so we need to add the probabilities together.
        prob[new_state] += (1 - s) * (1 - r)

    # Make sure that the probabilities sum to one
    assert (prob.sum() - 1) < np.spacing(1)
    return prob
{% endhighlight %}

### Reward vectors

The other important part of an MDP that we haven't discussed yet is the rewards.
The rewards depend on the state of the system.

... simple example...

{% highlight python %}
R = np.array([[ 5, 10],
              [-1,  2]])
{% endhighlight %}

Defind in [Possingham and Tuck (1997)][poss-tuck-97] to be zero if the population is extinct and one if the population is extant.
Therefore the rewards can be defined as a vector of length <math>S</math> (which is one of the valid ways of specifying rewards to PyMDPtoolbox).

### Putting it all together

Now we loop over the states and actions, getting the transition probabilities and fill in the transition probability matrix.

{% highlight python %}
def get_transition_and_reward_arrays(s):
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
    assert 0 <= s <= 1, "'s' must be between 0 and 1 not %f" % s
    # The transition probability array
    P = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    R = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Get the state index as inputs to our functions
        x, F = convert_index_to_state(idx)
        # The reward for being in this state is 1 if the population is extant
        if x != 0:
            R[idx] = 1
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            P[a][idx] = get_transition_probabilities(s, x, F, a)
    return (P, R)
{% endhighlight %}

What do the transition and reward arrays look like?...

## Solving a Markov decision process

The hardest part is over and now we want to solve the MDP.
Fortunately with PyMDPtoolbox, solving an MDP is very straightforward.
The minimum amount of work that you have to do is choose which algorithm you want to use, and decide on a discount factor.
In this case Possingham and Tuck mention in Section 3 that they are using a back-stepping method with a timeframe of 50 years, so the appropriate PyMDPtoolbox class to use would be `mdptoolbox.mdp.FiniteHorizon` with the number of periods equal to 50.
The second item to decide upon is the discount factor.
The discount factor is the discount applied to future rewards.
It's not clear from the paper if [Possingham and Tuck (1997)][poss-tuck-97] used a discount factor or not, but I will set it to 0.96 which means that future rewards have a discount rate of 4%.

{% highlight python %}
def solve_mdp():
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
    P, R = get_transition_and_reward_arrays(0.5)
    sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
    sdp.run()
    return sdp
{% endhighlight %}

I have added a printing function that prints the policy as a table, which is replicated in the table below.
It shows for each population abundance, as the rows, and years since fire, as the columns, which action should be chosen.
The forest patch should not be burned until the population is in the highest abundance class and the time since last fire is seven years.
This strategy would result in a cycle of burning the forest at (6, 7) which will cause the state to move to (5, 0), then do no action as the state transitions through (6, 1), (6, 2)...(6, 7) and then burn again thereby restarting the cycle.

{% highlight python %}
def print_policy(policy):
    """Print out a policy vector as a table to console

    Let ``S`` = number of states.

    The output is a table that has the population class as rows, and the years
    since a fire as the columns. The items in the table are the optimal action
    for that population class and years since fire combination.

    Parameters
    ----------
    p : array
        ``p`` is a numpy array of length ``S``.

    """
    p = np.array(policy).reshape(POPULATION_CLASSES, FIRE_CLASSES)
    print("    " + " ".join("%2d" % f for f in range(FIRE_CLASSES)))
    print("    " + "---" * FIRE_CLASSES)
    for x in range(POPULATION_CLASSES):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                     range(FIRE_CLASSES)))
{% endhighlight %}

Finally, we can run solve the MDP and have a look at the optimal policy.
The policy created by the finite horizon algorithm is an optimal policy for each period under consideration, it has the states as rows and each period as columns.
Possingham and Tuck consider policies 50 years from the terminal time, which represents a good long-term strategy.
To get this policy we take a slice from the policy array at the first column.

{% highlight python %}
sdp = solve_mdp()
print_policy(sdp.policy[:, 0])
{% endhighlight %}

Output:

<pre>
     0  1  2  3  4  5  6  7  8  9 10 11 12
    ---------------------------------------
  0| 0  0  0  0  0  0  0  0  0  0  0  0  0
  1| 0  0  0  0  0  0  0  0  0  0  0  0  0
  2| 0  0  0  0  0  0  0  0  0  0  0  0  0
  3| 0  0  0  0  0  0  0  0  0  0  0  0  0
  4| 0  0  0  0  0  0  0  0  0  0  0  0  0
  5| 0  0  0  0  0  0  0  0  0  0  0  0  0
  6| 0  0  0  0  0  0  0  1  1  1  1  1  1
</pre>

[poss-tuck-97]: http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf

### Reference
> Possingham H    & Tuck G, 1997, ‘Application of stochastic
dynamic programming to optimal fire management of a spatially structured
threatened species’, *MODSIM 1997*, vol. 2, pp. 813–817.

