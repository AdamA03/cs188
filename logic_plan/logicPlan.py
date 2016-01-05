# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game
import itertools


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    notA = ~A
    notB = ~B
    A_or_B = A | B
    notB_or_C = logic.disjoin((notB), (C))
    notA_iff_notB_or_C = notA % notB_or_C
    notA_or_notB_or_C = logic.disjoin((notA), (notB), (C))

    return logic.conjoin((A_or_B), (notA_iff_notB_or_C), (notA_or_notB_or_C))

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"

    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    notB = ~B
    notD = ~D
    notC = ~C
    B_or_D = logic.disjoin((B), (D))
    notB_and_notD = logic.conjoin((notB), (notD))
    C_iff_B_or_D = C % B_or_D  
    A_implies_notB_and_notD = A >> notB_and_notD 
    B_and_notC = logic.conjoin((B), (notC)) 
    not_B_and_notC_implies_A = ~B_and_notC >> A 
    notD_implies_C = notD >> C 

    return logic.conjoin((C_iff_B_or_D), (A_implies_notB_and_notD), (not_B_and_notC_implies_A), (notD_implies_C))

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    alive1 = logic.PropSymbolExpr('WumpusAlive', 1)
    alive0 = logic.PropSymbolExpr('WumpusAlive', 0)
    born0 = logic.PropSymbolExpr('WumpusBorn', 0)
    killed0 = logic.PropSymbolExpr('WumpusKilled', 0)

    return logic.conjoin((alive1 % logic.disjoin((logic.conjoin((alive0), ~(killed0))), (logic.conjoin(~(alive0), (born0))))), (~(alive0 & born0)), (born0))


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"

    return logic.disjoin(literals)


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"

    args = []
    for x, y in itertools.combinations(literals, 2):
        x = ~x
        y = ~y
        args.append((logic.disjoin(x,y)))
    return logic.conjoin(args)


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    #just call conjoin on at mostone and atleastone
    at_most = atMostOne(literals)
    at_least = atLeastOne(literals)
    return logic.conjoin(at_most, at_least)


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"

    list_of_action_time = []
    list_of_action = []
    for key, value in model.iteritems():
        action, time = logic.PropSymbolExpr.parseExpr(key)

        if value == True and action in actions:
            list_of_action_time.append((action, int(time)))
            list_of_action.append('')
    
    for action in list_of_action_time:
        list_of_action[action[1]] = action[0]
    return list_of_action



def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    left = walls_grid[x-1][y]
    top = walls_grid[x][y+1]
    right = walls_grid[x+1][y]
    bottom = walls_grid[x][y-1]
    
    possible_locations = []
   
    left_prev = logic.PropSymbolExpr(pacman_str, x-1, y, t-1)
    top_prev = logic.PropSymbolExpr(pacman_str, x, y+1, t-1)
    bottom_prev = logic.PropSymbolExpr(pacman_str, x, y-1, t-1)
    right_prev = logic.PropSymbolExpr(pacman_str, x+1, y, t-1)

    if not left:
        east_action = logic.PropSymbolExpr('East', t-1)
        possible_locations.append(logic.conjoin(left_prev, east_action))

    if not top:
        south_action = logic.PropSymbolExpr('South', t-1)
        possible_locations.append(logic.conjoin(top_prev, south_action))

    if not bottom:
        north_action = logic.PropSymbolExpr('North', t-1)
        possible_locations.append(logic.conjoin(bottom_prev, north_action))

    if not right:
        west_action = logic.PropSymbolExpr('West', t-1)
        possible_locations.append(logic.conjoin(right_prev, west_action))

    return logic.PropSymbolExpr(pacman_str, x, y, t) % logic.disjoin(possible_locations)

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    startx, starty = problem.getStartState()
    goalx, goaly = problem.getGoalState()
    pacman_start = logic.PropSymbolExpr(pacman_str, startx, starty, 0)
    all_actions = ['North', 'South', 'East', 'West']
    time = 0
    kb = pacman_start
    while time <= 50:
        goal_state = logic.PropSymbolExpr(pacman_str, goalx, goaly, time+1)
        actions_at_t = []
        actions_at_t.append(logic.PropSymbolExpr('North', time))
        actions_at_t.append(logic.PropSymbolExpr('West', time))
        actions_at_t.append(logic.PropSymbolExpr('East', time))
        actions_at_t.append(logic.PropSymbolExpr('South', time))

        only_one_action = exactlyOne(actions_at_t)
        kb = only_one_action & kb
        
        for x in range(1, width+1):
            for y in range(1, height+1):
                if not walls[x][y]:
                    if time == 0:
                        if x != startx or y != starty:
                            not_start = ~(logic.PropSymbolExpr(pacman_str, x, y, time))
                            kb = kb & not_start
                    ssa = pacmanSuccessorStateAxioms(x, y, time+1, walls)
                    kb = kb & ssa
        goal = kb & goal_state
        check = False
        if time >= (abs(startx - goalx) + abs(starty - goaly)) - 1:
            check = findModel(goal)
        if check:
            return extractActionSequence(check, all_actions)
        time += 1

     


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    startx, starty= problem.getStartState()[0]
    food_grid = problem.getStartState()[1]

    pacman_start = logic.PropSymbolExpr(pacman_str, startx, starty, 0)
    all_actions = ['North', 'South', 'East', 'West']
    time = 0
    kb = pacman_start
    food_loc = []
    while time <= 50:
        actions_at_t = []
        actions_at_t.append(logic.PropSymbolExpr('North', time))
        actions_at_t.append(logic.PropSymbolExpr('West', time))
        actions_at_t.append(logic.PropSymbolExpr('East', time))
        actions_at_t.append(logic.PropSymbolExpr('South', time))

        only_one_action = exactlyOne(actions_at_t)
        kb = only_one_action & kb
        food_loc = []
        check = pacman_start
        for x in range(1, width+1):
            for y in range(1, height+1):
                if not walls[x][y]:
                    if time == 0:
                        if x != startx or y != starty:
                            not_start = ~(logic.PropSymbolExpr(pacman_str, x, y, time))
                            kb = kb & not_start
                    ssa = pacmanSuccessorStateAxioms(x, y, time+1, walls)
                    kb = kb & ssa
                if food_grid[x][y]:
                    for t in range(0, time + 2):
                        food_loc.append(logic.PropSymbolExpr(pacman_str, x, y, t))

                    dis_food_loc = logic.disjoin(food_loc)
                    check = check & dis_food_loc
                    food_loc = []
        check = kb & check
        model = findModel(check)
        if model:
            return extractActionSequence(model, all_actions)
        time += 1
        check = pacman_start


def ghostPositionSuccessorStateAxioms(x, y, t, ghost_num, walls_grid):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    GE is going east, ~GE is going west 
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"

    left = walls_grid[x-1][y]
    right = walls_grid[x+1][y]
    
    possible_locations = []
   
    left_prev = logic.PropSymbolExpr(pos_str, x-1, y, t-1)
    right_prev = logic.PropSymbolExpr(pos_str, x+1, y, t-1)
    west_action = ~(logic.PropSymbolExpr(east_str, t-1))
    east_action = logic.PropSymbolExpr(east_str, t-1)

    if left and right:
        return logic.PropSymbolExpr(pos_str, x, y, t) % logic.PropSymbolExpr(pos_str, x, y, t-1) 

    if not left:
        possible_locations.append(logic.conjoin(left_prev, east_action))
    
    if not right:
        possible_locations.append(logic.conjoin(right_prev, west_action))

    return logic.PropSymbolExpr(pos_str, x, y, t) % logic.disjoin(possible_locations) 

def ghostDirectionSuccessorStateAxioms(t, ghost_num, blocked_west_positions, blocked_east_positions):
    """
    Successor state axiom for patrolling ghost direction state (t) (from t-1).
    west or east walls.
    Current <==> (causes to stay) | (causes of current)
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"
    

    west_action = ~(logic.PropSymbolExpr(east_str, t-1))
    east_action = logic.PropSymbolExpr(east_str, t-1)

    west_list = []
    east_list = []

    for pos in blocked_east_positions:
        x,y = pos
        not_blocked_east = ~logic.PropSymbolExpr(pos_str, x, y, t)
        lst = logic.conjoin(east_action, not_blocked_east)
        east_list.append(lst)

    for pos in blocked_west_positions:
        x,y = pos
        blocked_west = logic.PropSymbolExpr(pos_str, x, y, t)
        lst = logic.conjoin(west_action, blocked_west)
        west_list.append(lst)

    east_list = logic.conjoin(east_list)
    west_list = logic.disjoin(west_list)

    return logic.PropSymbolExpr(east_str, t) % logic.disjoin(east_list, west_list)


def pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    """
    ghost_strs = [ghost_pos_str+str(ghost_num) for ghost_num in xrange(num_ghosts)]

    "*** YOUR CODE HERE ***"

    no_ghost = []

    for ghost in ghost_strs:
        ghost_position_before = ~logic.PropSymbolExpr(ghost, x, y, t-1)
        no_ghost.append(ghost_position_before)
        ghost_position_now = ~logic.PropSymbolExpr(ghost, x, y, t)
        no_ghost.append(ghost_position_now)

    no_ghost_here = logic.conjoin(no_ghost)
    ghost_exists = ~(logic.conjoin(no_ghost))
    pacman_alive_prev = logic.PropSymbolExpr(pacman_alive_str, t-1)
    pacman_not_there = ~logic.PropSymbolExpr(pacman_str, x, y, t)

    return logic.PropSymbolExpr(pacman_alive_str, t) % (pacman_alive_prev & ((no_ghost_here) | (ghost_exists & pacman_not_there)))

def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostPlanningProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"

    startx, starty = problem.getStartState()[0]
    food_grid = problem.getStartState()[1]

    pacman_start = logic.PropSymbolExpr(pacman_str, startx, starty, 0)
    all_actions = ['North', 'South', 'East', 'West']
    time = 0 
    kb = pacman_start
    food_loc = []
    prev_ghost_pos = []
    curr_ghost_pos = []

    def nextGhostState(x, y, direction, t):
        x_y_d_t = []
        if direction == 'East':
            if walls[x+1][y]:
                x_y_d_t = [x-1, y, 'West', t+1]
            else:
                x_y_d_t = [x+1, y, 'East', t+1]
        else:
            if walls[x-1][y]:
                x_y_d_t = [x+1, y, 'East', t+1]
            else:
                x_y_d_t = [x-1, y, 'West', t+1]
        return x_y_d_t
    
    for ghost in problem.getGhostStartStates():
        gx, gy = ghost.getPosition()
        prev_ghost_pos.append([gx, gy, 'East', 0])

    while time <= 50:
        actions_at_t = []
        actions_at_t.append(logic.PropSymbolExpr('North', time))
        actions_at_t.append(logic.PropSymbolExpr('West', time))
        actions_at_t.append(logic.PropSymbolExpr('East', time))
        actions_at_t.append(logic.PropSymbolExpr('South', time))

        only_one_action = exactlyOne(actions_at_t)
        kb = only_one_action & kb
        food_loc = []
        check = pacman_start
        
        curr_ghost_pos = []
        i=0
        for pos in prev_ghost_pos:
            curr_ghost_pos.append(nextGhostState(pos[0], pos[1], pos[2], pos[3]))
            pacman_not_at_prev = ~logic.PropSymbolExpr(pacman_str, pos[0], pos[1], time + 1)
            pacman_not_at_curr = ~logic.PropSymbolExpr(pacman_str, curr_ghost_pos[i][0], curr_ghost_pos[i][1], time + 1)
            kb = kb & pacman_not_at_prev
            kb = kb & pacman_not_at_curr
            i += 1

        prev_ghost_pos = curr_ghost_pos
        
        
        for x in range(1, width+1):
            for y in range(1, height+1):
                
                if not walls[x][y]:
                    if time == 0:
                        if x != startx or y != starty:
                            not_start = ~(logic.PropSymbolExpr(pacman_str, x, y, time))
                            kb = kb & not_start
                    ssa = pacmanSuccessorStateAxioms(x, y, time+1, walls)
                    kb = kb & ssa
                if food_grid[x][y]:
                    for t in range(0, time + 2):
                        food_loc.append(logic.PropSymbolExpr(pacman_str, x, y, t))

                    dis_food_loc = logic.disjoin(food_loc)
                    check = check & dis_food_loc
                    food_loc = []
        check = kb & check
        model = findModel(check)
        if model:
            return extractActionSequence(model, all_actions)
        time += 1
        check = pacman_start
 

# Abbreviationsx
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
