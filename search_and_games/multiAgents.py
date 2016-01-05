# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # successorGameState = currentGameState.generatePacmanSuccessor(action)
        # newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def manhattanHeuristic(xy1, xy2):     
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = 0 
        newx, newy = newPos
        closestFood = []

        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            if newPos == ghostPos:
                return -9999
            else:
                dis = min(4, manhattanHeuristic(newPos, ghostPos))
                inverse = 1000*dis
                score += inverse

        if len(newFood.asList()) != 0:
            for foodPos in newFood.asList():
                fx, fy = foodPos
                dis = manhattanHeuristic(newPos, foodPos)
                closestFood.append(dis)
                minFood = min(closestFood)
                inverse = 1.0/minFood
                score += inverse
        return 1000*successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        bestVal = [-float("inf"), Directions.NORTH]

        v = self.max_value(gameState, 0, depth)
        return v[1]

    def max_value(self, gameState, current_agent, depth):
      """ Helper function that maximizes Pacman's score"""
      bestVal = [-float('inf'), Directions.NORTH]
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return [self.evaluationFunction(gameState)]
      for action in gameState.getLegalActions(current_agent):
        state = gameState.generateSuccessor(current_agent, action)
        v = max(bestVal[0], self.min_value(state, current_agent + 1, depth)[0])
        if v > bestVal[0]:
          bestVal = [v, action]
      return bestVal

    def min_value(self, gameState, current_agent, depth):
      """ Helper function that minimizes the ghost's score """ 
      bestVal = [float('inf'), Directions.NORTH]
      number_of_agents = gameState.getNumAgents()
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return [self.evaluationFunction(gameState)]      
      for action in gameState.getLegalActions(current_agent):
        state = gameState.generateSuccessor(current_agent, action)
        if current_agent < number_of_agents-1:
          v = self.min_value(state, current_agent+1, depth)[0]
        else:
          v = self.max_value(state, 0, depth - 1)[0]
        if v < bestVal[0]:
          bestVal = [v, action]
      return bestVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        v = self.max_value(gameState, 0, depth)
        return v[1]

    def max_value(self, gameState, current_agent, depth):
      """ Helper function that maximizes Pacman's score"""

      bestVal = [-float('inf'), Directions.NORTH]
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return [self.evaluationFunction(gameState)]
      for action in gameState.getLegalActions(current_agent):
        state = gameState.generateSuccessor(current_agent, action)
        v = max(bestVal[0], self.expectation_value(state, current_agent + 1, depth)[0])
        if v > bestVal[0]:
          bestVal = [v, action]
      return bestVal

    def expectation_value(self, gameState, current_agent, depth):
      """ Helper function that averages the ghost's score """
      average = 0.0
      average_results = []
      bestVal = [float('inf'), Directions.NORTH]
      number_of_agents = gameState.getNumAgents()
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return [self.evaluationFunction(gameState)]      
      for action in gameState.getLegalActions(current_agent):
        state = gameState.generateSuccessor(current_agent, action)
        if current_agent < number_of_agents-1:
          v = self.expectation_value(state, current_agent+1, depth)[0]
        else:
          v = self.max_value(state, 0, depth - 1)[0]
        average_results.append(v)
        average = min_average(average_results)
      return [average]


def min_average(list):
    """Helper function that will get the average """
    num = 0.0
    for i in list:
        num += i
        average_num = num / len(list)
    return float(average_num)


def mazeDistance(point1, point2, gameState):
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).

      DESCRIPTION: 
      We have a variable called score, which simply stores a value that is calculated based on the distance between
      pacman and each ghost and the distance between pacman and the closest food pellet. This value is then added to the 
      currentGameState's score at the very end. 
      If none of the ghosts are scared, pacman will take into account the distance between itself and each ghost.
      The closer the ghost, the lower the value that will be assigned to the variable 'score.' (we set a cap
      of 4 so that the value will be lower when a ghost is within 4 squares away from pacman. We also scaled up the score by 1000.
      We also add the inverse value of the closest food to the score.




    """

    def mazeDistance(point1, point2, gameState):
        x1, y1 = point1
        x2, y2 = point2
        walls = gameState.getWalls()
        assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
        assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
        prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
        return len(search.bfs(prob))

    def manhattanHeuristic(xy1, xy2):     
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = 0 
    x, y = pacmanPos
    closestFood = []

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        if 0 not in scaredTimes:
          if pacmanPos == ghostPos:
              return -9999
          else:
              dis = min(4, manhattanHeuristic(pacmanPos, ghostPos))
              inverse = 1000*dis
              score += inverse

    if len(foodGrid.asList()) != 0:
        for foodPos in foodGrid.asList():
            fx, fy = foodPos
            dis = manhattanHeuristic(pacmanPos, foodPos)
            closestFood.append(dis)

            minFood = min(closestFood)
            inverse = 1.0/minFood
            score += inverse
    return 1000*currentGameState.getScore() + score
        
better = betterEvaluationFunction

