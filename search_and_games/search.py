# search.py
# ---------
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


# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """ We wrote this class to store information about each state """
    def __init__(self, parentNode, currState, depth, action, path_cost): 
        self.parent_node = parentNode
        self.curr_state = currState
        self.depth = depth
        self.action = action
        self.path_cost = path_cost

    def getPathCost(self):
        return self.path_cost

    def getAction(self):
        return self.action

    def getParentNode(self):
        return self.parent_node

    def getCurrState(self):
        return self.curr_state

    def getDepth(self):
        return self.depth

    def addDepth(self, num):
        self.depth += num


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    currNode = Node(None, problem.getStartState(), 0, '', 0)
    visited, queue_set, queue = [], set(), util.Queue()
    queue.push(currNode)
    queue_set.add(currNode.getCurrState())

    while queue.isEmpty() != True:

        vertex = queue.pop()
        
        queue_set.remove(vertex.getCurrState())
        visited.append(vertex.getCurrState())

        if problem.goalTest(vertex.getCurrState()):
            return returnedPath(vertex)

        else:
            for action in problem.getActions(vertex.getCurrState()):
                child = Node(vertex, problem.getResult(vertex.getCurrState(), action), vertex.getDepth() + 1, action, 0)
                if problem.goalTest(child.getCurrState()):
                    return returnedPath(child)
                else:
                    if child.getCurrState() not in queue_set and child.getCurrState() not in visited:
                        queue.push(child)
                        queue_set.add(child.getCurrState())


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def returnedPath(node):
    """ This is our helper function that, given a node, returns the path """
    returned_path = []
    while node.parent_node:
        action = node.getAction()
        node = node.getParentNode()
        returned_path.append(action)
    returned_path.reverse()
    return returned_path

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    "*** YOUR CODE HERE ***"
    stack, stack_set = util.Stack(), set()
    limit = 0
    currNode = Node(None, problem.getStartState(), 0, '', 0)
    while True:
        result = dfs(problem, currNode, limit, stack, stack_set)
        limit += 1
        if result != None:
            return result
          
def dfs(problem, node, limit, stack, stack_set):
    """ This is our helper function used in iterativeDeepeningSearch """

    stack.push(node)
    stack_set.add(node.getCurrState())
    visited = []
    while stack.isEmpty() != True:
        vertex = stack.pop()
        stack_set.remove(vertex.getCurrState())
        visited.append(vertex.getCurrState())
        if problem.goalTest(vertex.getCurrState()):
            return returnedPath(vertex)

        else:

            if vertex.getDepth() <= limit:
                for action in problem.getActions(vertex.getCurrState()):
                    child = Node(vertex, problem.getResult(vertex.getCurrState(), action), vertex.getDepth() + 1, action, 0)
                    if problem.goalTest(child.getCurrState()):
                        return returnedPath(child)
                    else:
                        if child.getCurrState() not in stack_set and child.getCurrState() not in visited:
                            stack.push(child)
                            stack_set.add(child.getCurrState())

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = Node(None, problem.getStartState(), None, '', 0)
    frontier = util.PriorityQueue()
    frontier_states = []
    frontier.push(node, 0)
    frontier_states.append(node.getCurrState())
    explored = []
    while 1:
        if frontier.isEmpty():
            return 'failure'
        node = frontier.pop()
        node_state = node.getCurrState()
        frontier_states.remove(node_state)
        if problem.goalTest(node_state):
            return returnedPath(node)
        explored.append(node_state)

        for action in problem.getActions(node_state):
            cost_of_path = problem.getCost(node_state, action) + node.getPathCost()
            child = Node(node, problem.getResult(node_state, action), None, action, cost_of_path)
            child_state = child.getCurrState()
            if child_state not in explored and child_state not in frontier_states:
                frontier.push(child, child.getPathCost() + heuristic(child_state, problem))
                frontier_states.append(child_state)
            elif child_state in frontier_states:
                elem = frontier.pop()
                frontier_states.remove(elem.getCurrState())
                if child.getPathCost() + heuristic(child.getCurrState(), problem) < elem.getPathCost() + heuristic(elem.getCurrState(), problem):
                    frontier.push(child, child.getPathCost() + heuristic(child.getCurrState(), problem))
                    frontier_states.append(child_state)
                else:
                    frontier.push(elem, elem.getPathCost() + heuristic(elem.getCurrState(), problem))
                    frontier_states.append(elem.getCurrState())


# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
