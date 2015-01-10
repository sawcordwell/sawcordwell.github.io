---
layout: post
title: Optimal fire management of a threatened species, part 1
subtitle: Python MDP Toolbox worked example
date: 2015-01-10 23:30:00
categories: mdp conservation
tags: [MDP, Python, Markovian, decision theory, toolbox, tutorial, forest, conservation, fire]
---

The paper by [Possingham and Tuck (1997)][poss-tuck-97] was among the first to apply Markov decision theory to a conservation biology problem.
Here I will follow their paper and replicate their results as a worked example of how to use PyMDPtoolbox, a [Markov decison process (MDP) toolbox](<http://www.inra.fr/mia/T/MDPtoolbox/>) for Python.
For an introduction to MDPs see [Marescot _et al_ (2013)](<http://dx.doi.org/10.1111/2041-210X.12082>).

The complete source code for this tuorial is available in a [GitHub gist](https://gist.github.com/sawcordwell/bccdf42fcc4e024d394b).

## Setup

The first step is to install the PyMDPtoolbox package.
The repository for PyMDPtoolbox is hosted on GitHub at <https://github.com/sawcordwell/pymdptoolbox>, where you can find more detailed installation instructions.
Briefly you can install from [PyPI](<http://pypi.python.org>) with the `pip install pymdptoolbox` command or clone the repository with `git clone https://github.com/sawcordwell/pymdptoolbox.git`.
If you do not have [pip](https://pypi.python.org/pypi/pip) or [Git](http://git-scm.com/) installed, then I recommend that you install both.
If you still do not want to go to the trouble then download the zipped archive from [PyMDPtoolbox's GitHub page](https://github.com/sawcordwell/pymdptoolbox).
If you downloaded the zip archive then unzip it and enter the [setuptools](http://docs.python.org/2/library/distutils.html) command `python setup.py install` into a console in pymdptoolbox's top-level directory.
Now you should be able to `from mdptoolbox import mdp` in the Python console.

First, we need to download the [paper][poss-tuck-97] and read through Section&nbsp;1 and Section&nbsp;2.
Okay, now that we know what the problem is we can get to work.
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
They do not specifically mention how many classes there are to represent the years since last fire, but judging by Figure&nbsp;3 there are 13 fire classes (`FIRE_CLASSES`).
A state is made up of a population class component and a years since fire class component, so the number of states (`STATES`) is the number of both classes multiplied together.
In Possingham and Tuck there are four actions, but since I am only considering a single population for now that means I only have two actions (`ACTIONS`).
Action `0` is do nothing, and action `1` is burn the forest patch.
These constants are strictly neccesary but it leaves reading the code much nicer.

{% highlight python %}
# The number of population abundance classes
POPULATION_CLASSES = 7
# The number of years since a fire classes
FIRE_CLASSES = 13
# The number of states
STATES = POPULATION_CLASSES * FIRE_CLASSES
# The number of actions
ACTIONS = 2
ACTION_NOTHING = 0
ACTION_BURN = 1
{% endhighlight %}

## Input validation

Next we create some functions to help validate the inputs to various functions that we will define.
We're going to want to send the action, population class, fire class, and a probability as input to multiple functions.
If the values do not make sense, then an exception is raised that prints a useful error message that tells you what inout was expected and what input was passed.

{% highlight python %}
def check_action(x):
    """Check that the action is in the valid range."""
    if not (0 <= x < ACTIONS):
        msg = "Invalid action '%s', it should be in {0, 1}." % str(x)
        raise ValueError(msg)

def check_population_class(x):
    """Check that the population abundance class is in the valid range."""
    if not (0 <= x < POPULATION_CLASSES):
        msg = "Invalid population class '%s', it should be in {0, 1, …, %d}." \
              % (str(x), POPULATION_CLASSES - 1)
        raise ValueError(msg)

def check_fire_class(x):
    """Check that the time in years since last fire is in the valid range."""
    if not (0 <= x < FIRE_CLASSES):
        msg = "Invalid fire class '%s', it should be in {0, 1, …, %d}." % \
              (str(x), FIRE_CLASSES - 1)
        raise ValueError(msg)

def check_probability(x, name="probability"):
    """Check that a probability is between 0 and 1."""
    if not (0 <= x <= 1):
        msg = "Invalid %s '%s', it must be in [0, 1]." % (name, str(x))
        raise ValueError(msg)
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
        The time in years since last fire.

    Returns
    -------
    r : float
        The habitat suitability.

    """
    if years < 0:
        msg = "Invalid years '%s', it should be positive." % str(years)
        raise ValueError(msg)
    if years <= 5:
        return 0.2*years
    elif 5 <= years <= 10:
        return -0.1*years + 1.5
    else:
        return 0.5
{% endhighlight %}

## States, actions, transitions and rewards

The next part of the problem is how to define the states, actions and rewards.
The way that PyMDPtoolbox handles states is by indexing into matrices: each row represents a state that the system can start in, and the columns represent the states that the system can transition to.
Index numbers refer to the same state in both rows and columns, and the value of the matrix in the element represents the transition probability.
So for example given row 1, column 2, with a value of 0.5 then this means that the probability of transitioning to state 2 from state 1 is 0.5.
This matrix is referred to as the _transition probability matrix_.
There is one transition probability matrix for each action.
So, with the value at row <math>i</math>, column <math>j</math> of matrix <math>a</math>, then we know the probability of transitioning from state <math>i</math> to state <math>j</math> given action <math>a</math> is taken.

While there are a number of possible alternatives to storing transition probabilities (such as a Python `dict` or a custom class), the way that PyMDPtoolbox uses them is through [NumPy](<http://www.numpy.org>) arrays.
A single array will hold all the transition information for all the actions.
The toolbox's convention is to given the name `P` to the transition probability array, and that is what we will use here.

The array needs to be specified in a certain way.
It should have 3 dimensions, where the first dimension corresponds to the actions and the second dimension corresponds to the initial states before any transition, and the thrid dimension corresponds to the next states after any transition.
Each element of the array stores a probability: the probability that the state of the system transitions from the initial state <math>s</math> to the next state <math>s'</math> given that action <math>a</math> was taken.
Let the total number of actions be <math>A</math> and the total number of states be <math>S</math>, then ``P`` is going to be an array with size <math>A × S × S</math>.
Each slice of the array along the first dimenion is an <math>S × S</math> matrix, which is the transition probability matrix for the corresponding action.

The rewards are also stored as NumPy arrays.
Briefly, rewards indicate how desirable each state is to be in, and the MDP algorithm's goal is to maximise rewards over the long run.
The reward `R` can be either a vector of length <math>S</math> or a numpy array of size <math>S × A</math>.
The difference is that you can specify rewards as a function of state, or as a function of state-action pairs, whichever suits your needs better.
For this example, we only need to specify reward as a function of state.

### Probability transition matrices

Let us look at a simple example of a transition array for a hypothetical system before moving on.
In this example there are two states and two actions:

{% highlight python %}
P = np.array([[[0.5, 0.5],
               [0.8, 0.2]],
              [[0.0, 1.0],
               [0.1, 0.9]]])
{% endhighlight %}

Make sure you enter this into a Python session if you do not fully understand transition probabilities yet.
`P[0]` corresponds to the transition probability matrix for action `0`, and `P[1]` for action `1`.
If the system is in state `0` and action `0` is taken, then there is a 50% chance that the system will transition back to state `0` (i.e. stay in the same state) and a 50% chance that the system will transition to state `1`.
If action `1` is taken instead, then the system is guaranteed to transition to state `1`.
You will notice that each row sums to one, and this is a requirement for transition probability matrices.

So, each state is uniquely identified by an index into the transition probability matrix, but we think of the state as a combination of the population abundance class and the number of years since the last fire.
Therefore we need a way to translate from the human understandable state specified as a set of variables, to the machine readable form of an index and vice-versa.
These two functions will do the trick:

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
    check_population_class(population)
    check_fire_class(fire)
    return population*FIRE_CLASSES + fire
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
    if not (0 <= index < STATES):
        msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
              (str(index), STATES - 1)
        raise ValueError(msg)
    population = index // FIRE_CLASSES
    fire = index % FIRE_CLASSES
    return (population, fire)
{% endhighlight %}

Try testing them out and find which inputs will cause an error.

Now we need to move onto the logic of transitioning the states.
We will do it so that we can calculate the transition probabilities one state at a time.
This means that we will be filling in the transition probability matrix row by row.
Remember, any given transition can have a probability of zero, but each row of the transition probability matrix must sum to one; so with probability 1 any given state will transition to the next state (it can transition back to the same state).
The dynamics of the transition probabilities are given in Section 2.1 and Figure 1 of [Possingham and Tuck (1997)][poss-tuck-97].
We will need to set up a NumPy array to store the row of probabilities, and also get the habitat suitability based on the years since last fire <math>F</math>:

{% highlight python %}
...
prob = np.zeros(STATES)
r = get_habitat_suitability(F)
...
{% endhighlight %}

First we work out how to transition the years since last fire.
This is simple to describe: given action do nothing, <math>a = 0</math>, the years since last fire <math>F</math> will increase by one until it has reached the largest class, after which it is absorbed into the largest class; and given action burn, <math>a = 1</math>, the years since last fire <math>F</math> will go back to zero:

{% highlight python %}
def transition_fire_state(F, a):
    """Transition the years since last fire based on the action taken.

    Parameters
    ----------
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    F : int
        The time in years since last fire.

    """
    ## Efect of action on time in years since fire.
    if a == ACTION_NOTHING:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == ACTION_BURN:
        # When the patch is burned set the years since fire to 0.
        F = 0

    return F
{% endhighlight %}

Now we need work out how the transitions of the population abundance should work.
This is broken into three components:

1. When the abundance class is zero (extinct), <math>x = 0</math>;
2. When the abundance class is at its maximum, <math>x = 6</math>; and
3. When the abundance class is intermediate, <math>0 < x < 6</math>.

If the population is extinct, then it stays extinct, so:

{% highlight python %}
...
new_state = convert_state_to_index(0, F)
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
if a == ACTION_BURN:
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
Also, in this case the `transition_same` and `transition_down` are the same state, so the probabilities of these need to be summed:

{% highlight python %}
...
# Population abundance class can stay the same, transition up, or
# transition down.
transition_same = x
transition_up = x + 1
transition_down = x - 1
# If action 1 is taken, then the patch is burned so the population
# abundance moves down a class.
if a == ACTION_BURN:
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
        The class-independent probability of the population staying in its
        current population abundance class.
    x : int
        The population abundance class of the threatened species.
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    prob : array
        The transition probabilities as a vector from state (``x``, ``F``) to
        every other state given that action ``a`` is taken.

    """
    # Check that input is in range
    check_probability(s)
    check_population_class(x)
    check_fire_class(F)
    check_action(a)

    # a vector to store the transition probabilities
    prob = np.zeros(STATES)

    # the habitat suitability value
    r = get_habitat_suitability(F)
    F = transition_fire_state(F, a)

    ## Population transitions
    if x == 0:
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
        if a == ACTION_BURN:
            transition_same -= 1
            transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = 1 - (1 - s)*(1 - r)
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        prob[new_state] = (1 - s)*(1 - r)
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        transition_same = x
        transition_up = x + 1
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == ACTION_BURN:
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
        prob[new_state] = (1 - s)*r
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        # In the case when transition_down = 0 before the effect of an action
        # is applied, then the final state is going to be the same as that for
        # transition_same, so we need to add the probabilities together.
        prob[new_state] += (1 - s)*(1 - r)

    # Make sure that the probabilities sum to one
    assert (prob.sum() - 1) < np.spacing(1)
    return prob
{% endhighlight %}

### Reward vectors

Let us revisit rewards for a minute.
We already know that rewards depend on the state of the system, and they can also depend on the action taken.
A simple example similar to the transition example above is as follows:

{% highlight python %}
R = np.array([[ 5, 10],
              [-1,  2]])
{% endhighlight %}

There are two states and two actions.
This time the actions are the columns and the states are the rows, so the reward for being in state `0` and taking action `1` is 10.

Rewards are defind in [Possingham and Tuck (1997)][poss-tuck-97] to be zero if the population is extinct and one if the population is extant.
Therefore the rewards only need be defined as a vector of length <math>S</math>.

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
    check_probability(s)

    # The transition probability array
    transition = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    reward = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Get the state index as inputs to our functions
        x, F = convert_index_to_state(idx)
        # The reward for being in this state is 1 if the population is extant
        if x != 0:
            reward[idx] = 1
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            transition[a][idx] = get_transition_probabilities(s, x, F, a)

    return (transition, reward)
{% endhighlight %}

If you generate them like `P, R = get_transition_and_reward_arrays(0.5)` you will notice that `R` is a vector of length 91, and that `P` is an array with a shape `(2, 91, 91)`.
`P[0]` is the matrix corresponding to the transition probabilities for doing nothing, and `P[1]` is the matrix of transition probabilities for buring the forest.
You can verify that each row sums to one with `P.sum(2)`.

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
    sdp : mdptoolbox.mdp.FiniteHorizon
        The PyMDPtoolbox object that represents a finite horizon MDP. The
        optimal policy for each stage is accessed with mdp.policy, which is a
        numpy array with 50 columns (one for each stage).

    """
    P, R = get_transition_and_reward_arrays(0.5)
    sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
    sdp.run()
    return sdp
{% endhighlight %}

We can add a printing function that prints the policy as a table.
It shows for each population abundance, as the rows, and years since fire, as the columns, which action should be chosen.

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

The forest patch should not be burned until the population is in the highest abundance class and the time since last fire is seven years.
This strategy would result in a cycle of burning the forest at (6, 7) which will cause the state to move to (5, 0), then do no action as the state transitions through (6, 1), (6, 2)…(6, 7) and then burn again thereby restarting the cycle.

[poss-tuck-97]: http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf

### Reference
Possingham H    & Tuck G, 1997, ‘Application of stochastic
dynamic programming to optimal fire management of a spatially structured
threatened species’, *MODSIM 1997*, vol. 2, pp. 813–817.

