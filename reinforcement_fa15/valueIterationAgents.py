# valueIterationAgents.py
# -----------------------
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

from learningAgents import ValueEstimationAgent
import collections
import time

class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        # highestprob_of getting_next_state (   value_of_curr_state + (discount * value_of_next_state) )
        start = time.time()
        weighted_sum_all_actions = []
        i = 0
        while i <= iterations - 1:
            state = states[i%len(states)]
            if self.mdp.isTerminal(state):
                i += 1
            else:
                if len(self.mdp.getPossibleActions(state)) != 0:
                    for action in self.mdp.getPossibleActions(state):
                        tup_sum = self.getQValue(state, action)
                        weighted_sum_all_actions.append(tup_sum)
                    self.values[state] = max(weighted_sum_all_actions)
                    weighted_sum_all_actions = []
                    r = sum(abs(value - 100) for state, value in self.values.items() if not self.mdp.isTerminal(state))
                    
                    print("sum: " + str(r) + " , i: " + str(i))
                    print("time" + str(time.time() - start))
                    i += 1
     

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tup_sum = 0
        if self.mdp.isTerminal(state):
            return self.values[state]
        reward = self.mdp.getReward(state)
        prob = self.mdp.getTransitionStatesAndProbs(state, action)
        if len(prob) == 0:
            return self.values[state]
        for tup in prob:
            tup_sum += tup[1] * (reward + self.values[tup[0]] * self.discount)
        return tup_sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        action_dict = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            Q_value = self.getQValue(state, action)
            action_dict[action] = Q_value
        best_action = action_dict.argMax() 
        return best_action
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        start = time.time()

        predecessors_dict = {} #key = state, value = set_of_predecessors

        for state in states:
            predecessors_dict[state] = set()

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                resultingStates = self.mdp.getTransitionStatesAndProbs(state, action)
                
                for tup in resultingStates:
                    if tup[1] > 0:
                        predecessors_dict[tup[0]].add(state)

        p_queue = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state) != True:
                actionQval = []
                for action in self.mdp.getPossibleActions(state):
                    actionQval.append(self.getQValue(state, action))
                
                diff = abs(self.values[state] - max(actionQval))
                p_queue.push(state, -diff)

        i = 0
        while i <= self.iterations - 1:
            if p_queue.isEmpty():
                break;
            else:
                s = p_queue.pop()
                if self.mdp.isTerminal(s) != True:
                    actionQval = []
                    for action in self.mdp.getPossibleActions(s):
                        actionQval.append(self.getQValue(s, action))
                    self.values[s] = max(actionQval)

                    for p in predecessors_dict[s]:
                        p_actionQval = []
                        for action in self.mdp.getPossibleActions(p):
                            p_actionQval.append(self.getQValue(p, action))
                        diff = abs(self.values[p] - max(p_actionQval))

                        if diff > theta:
                            p_queue.update(p, -diff)

            r = sum(abs(value - 100) for state, value in self.values.items() if not self.mdp.isTerminal(state))
            print("sum: " + str(r) + " , i: " + str(i))
            print("time" + str(time.time() - start))
            i += 1
        
































