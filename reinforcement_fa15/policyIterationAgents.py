# policyIterationAgents.py
# ------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np

from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 20):
        """
          Your policy iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        states = self.mdp.getStates()
        # initialize policy arbitrarily
        self.policy = {}
        for state in states:
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
                self.policy[state] = self.mdp.getPossibleActions(state)[0]
        # initialize policyValues dict
        self.policyValues = {}
        for state in states:
            self.policyValues[state] = 0

        for i in range(self.iterations):
            # step 1: call policy evaluation to get state values under policy, updating self.policyValues
            self.runPolicyEvaluation()
            # step 2: call policy improvement, which updates self.policy
            self.runPolicyImprovement()

    def runPolicyEvaluation(self):
        """ Run policy evaluation to get the state values under self.policy. Should update self.policyValues.
        Implement this by solving a linear system of equations using numpy. """
        "*** YOUR CODE HERE ***"
        states = list(self.policy.keys())
        A = np.eye(len(states))
        b = np.zeros(len(states))

        for state in states:
            if self.mdp.isTerminal(state) != True:
                i = states.index(state)
                action = self.getPolicy(state)
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    j = states.index(nextState)
                    if i == j:
                        A[i][j] = 1 - self.discount * prob
                    else:
                        A[i][j] = -1 * self.discount * prob

        for state in states:
            if self.mdp.isTerminal(state) != True:
                sum_of_probs = 0
                i = states.index(state)
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        sum_of_probs += prob
                    b[i] = self.mdp.getReward(state) * sum_of_probs

        x = np.linalg.solve(A,b)

        for i in range(len(x)):
            self.policyValues[states[i]] = x[i]


    def runPolicyImprovement(self):
        """ Run policy improvement using self.policyValues. Should update self.policy. """
        "*** YOUR CODE HERE ***"
        states = list(self.policy.keys())

        for state in states:
            if self.mdp.isTerminal(state) != True:
                maxQVal = -float("inf")
                bestAction = None
                for action in self.mdp.getPossibleActions(state):
                    prob = self.computeQValueFromValues(state, action)
                    if prob > maxQVal:
                        maxQVal = prob
                        bestAction = action
                self.policy[state] = bestAction
            else:
                self.policy[state] = None


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.policyValues.
        """
        "*** YOUR CODE HERE ***"
        tup_sum = 0
        if self.mdp.isTerminal(state):
            return self.getValue(state)
        reward = self.mdp.getReward(state)
        prob = self.mdp.getTransitionStatesAndProbs(state, action)
        if len(prob) == 0:
            return self.getValue(state)
        for tup in prob:
            tup_sum += tup[1] * (reward + self.getValue(tup[0]) * self.discount)
        return tup_sum

    def getValue(self, state):
        return self.policyValues[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
