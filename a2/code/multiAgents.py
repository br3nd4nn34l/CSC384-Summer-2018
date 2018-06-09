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

from game import Directions
import random, util

from game import Agent, Actions


# region Maze Distance Calculation (copied from A1 and modified)

class SearchPath:
    def __init__(self, parent, edge, state):
        self.parent = parent
        self.edge = edge
        self.state = state

    def add_child(self, edge, value):
        return SearchPath(self, edge, value)

    def to_edges(self):
        """(SearchPath) -> [Edge]

        Converts the SearchPath into a list of Edges (that derive the state of the path).

        :param self: some SearchPath
        :return: list of Edge
        """
        edges = []

        cur = self
        while cur is not None:
            edges += [cur.edge]
            cur = cur.parent

        # Slice off the first element because it's going to be None
        return list(reversed(edges))[1:]

    def has_state(self, state):
        """(SearchPath, State) -> bool

        Returns true iff the SearchPath or one of it's ancestors' states is state.

        :param state: some state
        :return: boolean indicating whether the SearchPath contains state
        """
        cur = self

        while cur is not None:
            if cur.state == state:
                return True
            cur = cur.parent

        return False

    @staticmethod
    def root(value):
        return SearchPath(None, None, value)


class SearchProcedure(object):
    def __init__(self, problem):
        self.problem = problem

    def init_fringe(self):
        """(SearchProcedure) -> None
        Initializes the fringe for this search procedure.
        :return: None
        """
        util.raiseNotDefined()

    def fringe_is_empty(self):
        """(SearchProcedure) -> bool
        Returns whether or not the fringe is empty.
        :return: True iff the fringe is empty, False otherwise
        """
        util.raiseNotDefined()

    def push_path(self, path):
        """(SearchProcedure, SearchPath) -> None

        Pushes path into the fringe.

        :param path: the SearchPath to push into the fringe
        :return: None
        """
        util.raiseNotDefined()

    def pop_path(self):
        """(SearchProcedure) -> SearchPath

        Pops a path from the fringe and returns it.

        :return: the SearchPath popped from the fringe
        """

        util.raiseNotDefined()

    def is_goal(self, state):
        """(SearchProcedure, State) -> bool

        Returns whether or not state is the goal state for this search procedure.

        :param state: some state
        :return: True iff state is a goal state
        """
        return self.problem.isGoalState(state)

    def get_successors(self, path):
        return self.problem.getSuccessors(path.state)

    def get_state_of_successor(self, succ):
        return succ[0]

    def get_edge_of_successor(self, succ):
        util.raiseNotDefined()

    def get_action_of_edge(self, edge):
        util.raiseNotDefined()

    def run(self):

        # Initialize the fringe and push the start state into it
        self.init_fringe()
        self.push_path(SearchPath.root(self.problem.getStartState()))

        # Start draining the fringe (it's initialized with the start state path)
        while not self.fringe_is_empty():

            # Pop the path from the fringe
            popped_path = self.pop_path()

            # Return the answer if the path's state is the goal state
            if self.is_goal(popped_path.state):
                return [self.get_action_of_edge(x) for x in popped_path.to_edges()]

            # Otherwise get the successive states for the popped state,
            # and put each path into the fringe
            for successor in self.get_successors(popped_path):
                state = self.get_state_of_successor(successor)
                edge = self.get_edge_of_successor(successor)
                path = popped_path.add_child(edge, state)
                self.push_path(path)

        # Return an empty path by default
        return []


class CycleCheckedProcedure(SearchProcedure):
    """
    Partial implementation of SearchProblemProcedure
    that uses cycle checking in get_successors
    (states of each path must be unique ACROSS ALL PATHS)
    """

    def __init__(self, problem):
        super(CycleCheckedProcedure, self).__init__(problem)
        self.completed_states = set()
        self.open_states = set()

    def push_path_actual(self, path):
        util.raiseNotDefined()

    def push_path(self, path):

        # Push the path using the callback
        self.push_path_actual(path)

        # The path's state is now opened
        self.open_states.add(path.state)

    def pop_path_actual(self):
        util.raiseNotDefined()

    def pop_path(self):

        # Pop the path using the callback
        popped = self.pop_path_actual()

        # Remove the path's end state from the open set
        self.open_states.remove(popped.state)

        return popped

    def get_successors(self, path, allow_opened):
        ret = []

        # If we have completed the state, we can't expand it again
        if path.state in self.completed_states:
            return []

        # Gather each successor whose state is incomplete and unopened
        for successor in super(CycleCheckedProcedure, self).get_successors(path):
            state = self.get_state_of_successor(successor)

            # True when state is unopened or allow_opened is True
            is_unopened = (state not in self.open_states) or allow_opened

            if (state not in self.completed_states) and is_unopened:
                ret += [successor]

        # Mark the state as completed
        self.completed_states.add(path.state)

        return ret


class BfsProcedure(CycleCheckedProcedure):
    """
    Complete implementation of BFS as a CycleCheckedProcedure
    """

    def __init__(self, problem):
        CycleCheckedProcedure.__init__(self, problem)

    def init_fringe(self):
        self.fringe = util.Queue()

    def fringe_is_empty(self):
        return self.fringe.isEmpty()

    def push_path_actual(self, path):
        self.fringe.push(path)

    def pop_path_actual(self):
        return self.fringe.pop()

    # Cannot consider opened paths
    def get_successors(self, path):
        return super(BfsProcedure, self).get_successors(path, False)

    def get_edge_of_successor(self, succ):
        return succ[1]

    def get_action_of_edge(cls, edge):
        return edge


class MapDistanceProblem(object):
    """
    Light-weight version of the PositionSearchProblem.
    States are simply coordinates on the grid / map.
    """

    def __init__(self, walls, start_point, end_point):
        self.walls = walls
        self.start_point = start_point
        self.end_point = end_point

    def getStartState(self):
        return self.start_point

    def isGoalState(self, state):
        return state == self.end_point

    def getSuccessors(self, state):

        # Deserialize the state into coordinate components
        x, y = state

        # Determine successor states (actions that don't crash us into walls)
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nxt_x, nxt_y = int(x + dx), int(y + dy)

            if not self.walls[nxt_x][nxt_y]:
                next_state = (nxt_x, nxt_y)
                edge_val = 1
                successors += [(next_state, action, edge_val)]

        return successors


class MapDistanceCalculator(object):
    """
    Utilizes MapDistanceProblem and caching to
    quickly compute distances between points on the map.
    """

    def __init__(self, walls):
        self.walls = walls
        self.dist_cache = {}

    def _compute_moves(self, pt1, pt2):
        """
        Returns the moves needed to move from pt1 to pt2 on this PacMan map.
        """
        return BfsProcedure(MapDistanceProblem(self.walls, pt1, pt2)).run()

    def _compute_path_coords(self, pt1, pt2):
        """
        Returns the coordinates on the path made when
        the moves generated by compute_moves are followed
        """
        x1, y1 = pt1
        ret = [(x1, y1)]
        for move in self._compute_moves(pt1, pt2):
            dx, dy = Actions.directionToVector(move)
            cur_x, cur_y = ret[-1]
            nxt_x, nxt_y = int(cur_x + dx), int(cur_y + dy)
            ret += [(nxt_x, nxt_y)]
        return ret

    def __call__(self, pt1, pt2):
        """
        Returns the number of moves needed to travel between pt1 and pt2.

        Define notation: A~>B : shortest path from A to B

        For a graph with uniform edge costs,
        For every pair of coordinates (C1, C2) in pt1~>pt2,
        C1~>C2 must be a sub-path of pt1~>pt2

        Thus, after obtaining a complete path from one node to another,
        we can cache each sub-path of the obtained path
        for faster re-computation
        """
        # Unpack points
        point1 = tuple(int(v) for v in pt1)
        point2 = tuple(int(v) for v in pt2)
        x1, y1 = point1
        x2, y2 = point2

        # Either coordinate is a wall -> infinity
        if (self.walls[x1][y1] or self.walls[x2][y2]):
            return float('inf')

        # Coordinates are the same -> 0
        if point1 == point2:
            return 0

        # Coordinates are right next to each other -> 1
        if util.manhattanDistance(point1, point2) == 1:
            return 1

        # Reversed call may already be in cache
        if (point2, point1) in self.dist_cache:
            return self.dist_cache[(point2, point1)]

        # Fill the cache
        if not (point1, point2) in self.dist_cache:

            # Compute the coordinates of the path between pt1 and pt2
            path_coords = self._compute_path_coords(point1, point2)

            # Distance is infinite
            if path_coords == []:
                self.dist_cache[(point1, point2)] = float('inf')

            # Distances of each sub-path are simply differences in indices
            for i in range(len(path_coords)):
                for j in range(i + 1, len(path_coords)):
                    key = (path_coords[i], path_coords[j])
                    self.dist_cache[key] = j - i

        # Get the cached value, it has to be computed at some point
        return self.dist_cache[(point1, point2)]


class MapDistance(object):
    """
    Maintains a dictionary of {map : calculator}
    for fast distance calculations across different maps.
    """
    calculators = {}

    @classmethod
    def get_calculator(cls, walls):
        """
        Returns a MapDistanceCalculator, given walls.
        """
        if walls not in cls.calculators:
            cls.calculators[walls] = MapDistanceCalculator(walls)
        return cls.calculators[walls]


# endregion

# region Evaluation Function Feature Extraction

def distances(game_state, positions, distance):
    """
    Returns a list of distances between PacMan and positions.
    """
    pacman_pos = game_state.getPacmanPosition()
    return [distance(pacman_pos, pos) for pos in positions]

def fearless_ghost_positions(game_state):
    """
    Returns a list of positions of fearless ghosts in game_state.
    """
    ghost_states = game_state.getGhostStates()
    return [
        state.getPosition()
        for state in ghost_states
        if state.scaredTimer < 1
    ]

def feared_ghost_positions(game_state):
    """
    Returns a list of positions of feared ghosts in game_state.
    """
    ghost_states = game_state.getGhostStates()
    return [
        state.getPosition()
        for state in ghost_states
        if state.scaredTimer > 0
    ]

def worrisome_ghost_positions(game_state, detection_radius, distance):
    """
    Returns a list of positions of ghosts that
    are within detection_radius distance units of PacMan.
    """
    pacman = game_state.getPacmanPosition()
    return [
        position for position in fearless_ghost_positions(game_state)
        if distance(position, pacman) < detection_radius
    ]

def total_worrisome_distance(game_state, detection_radius, distance):
    """
    Returns the total distance between PacMan and all the worrisome ghosts in game_state.
    If there are no worrisome ghosts, default to detection_radius * number of fearless ghosts.
    """
    return sum(distances(
        game_state,
        worrisome_ghost_positions(game_state, detection_radius, distance),
        distance
    )) or (detection_radius * len(fearless_ghost_positions(game_state)))

def nearest_food_distance(game_state, distance, default=1):
    """
    Returns the distance between PacMan and the nearest piece of food.
    If there is no such food, return default.
    """
    return min(distances(game_state, game_state.getFood().asList(), distance) or [default])

def nearest_capsule_distance(game_state, distance, default=0):
    """
    Returns the distance between PacMan and the nearest capsule.
    If there is no capsule, return default.
    """
    return min(distances(game_state, game_state.getCapsules(), distance) or [default])

# endregion

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Get distance calculator
        map_distance = MapDistance.get_calculator(currentGameState.getWalls())

        # Difference in score (want to maximize)
        score_diff = successorGameState.getScore() - currentGameState.getScore()

        # Distance until nearest food (want to minimize -> invert for scoring)
        food_dist = nearest_food_distance(successorGameState, map_distance, default=1)

        # Total distance to each worrisome ghost (want to maximize)
        ghost_dist = total_worrisome_distance(successorGameState, 3, map_distance)

        # Weighing each component
        weighted = (score_diff, 5.0 / food_dist, 10 * ghost_dist)

        # Return the sum of the weighted components
        return sum(weighted)


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
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class AssistedMultiAgent(MultiAgentSearchAgent):
    """
    MultiAgent inheritor with various helper methods 
    that can be used to assist with programming the getAction method
    """

    def is_terminal(self, game_state, depth):
        """
        Return whether game_state is a win or loss, or depth is 0.
        """
        return depth == 0 or \
               game_state.isWin() or \
               game_state.isLose()

    def is_last_ghost(self, game_state, agent_index):
        """
        Returns whether or not agent_index represents the last ghost.
        """
        return agent_index == (game_state.getNumAgents() - 1)

    def is_pacman(self, agent_index):
        return agent_index == 0

    def get_next_agent(self, game_state, agent_index):
        """
        Gets the next agent index to play.
        """
        num_agents = game_state.getNumAgents()
        return (agent_index + 1) % num_agents

    def get_next_depth(self, game_state, agent_index, depth):
        """
        If agent_index is on the last ghost, return depth - 1.
        Otherwise, returns depth.
        """
        if self.is_last_ghost(game_state, agent_index):
            return depth - 1
        return depth

    def get_succeeded_actions(self, game_state, agent_index):
        return (
            (game_state.generateSuccessor(agent_index, action), action)
            for action
            in game_state.getLegalActions(agent_index)
        )

    def getAction(self, state):
        util.raiseNotDefined()


class MinimaxAgent(AssistedMultiAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, game_state, agent_index, depth):
        """
        Return pair of form (score, action), where:
            action = some legal move doable by agent_index from game_state
            score = score obtained by agent_index executing action
        """

        # Return if the node is terminal
        if self.is_terminal(game_state, depth):
            return (self.evaluationFunction(game_state), None)

        # Iterator of (minimax score, action) pairs
        next_agent = self.get_next_agent(game_state, agent_index)
        next_depth = self.get_next_depth(game_state, agent_index, depth)

        scored_actions = (
            (self.minimax(successor, next_agent, next_depth)[0], action)
            for (successor, action)
            in self.get_succeeded_actions(game_state, agent_index)
        )

        # Score is the first element of each tuple
        def get_score(tup):
            return tup[0]

        if self.is_pacman(agent_index):
            return max(scored_actions, key=get_score)
        else:
            return min(scored_actions, key=get_score)

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
        score, action = self.minimax(gameState, 0, self.depth)
        return action


class AlphaBetaAgent(AssistedMultiAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_pruned(self, game_state, agent_index, depth, alpha, beta):

        # Score is the first element of each tuple
        def get_score(tup):
            return tup[0]

        # Values for recursive call
        ret_action = None
        high_score = -float('inf')
        next_alpha = alpha
        next_agent = self.get_next_agent(game_state, agent_index)
        next_depth = self.get_next_depth(game_state, agent_index, depth)

        for (successor, action) in self.get_succeeded_actions(game_state, agent_index):

            # Update high score and the associated action
            high_score, ret_action = max([
                (high_score, ret_action),
                (self.alpha_beta(successor, next_agent, next_depth, next_alpha, beta)[0], action)
            ], key=get_score)

            # Update the alpha
            next_alpha = max(next_alpha, high_score)

            # No point in looking past here
            if beta <= next_alpha:
                break

        return (high_score, ret_action)

    def beta_pruned(self, game_state, agent_index, depth, alpha, beta):

        # Score is the first element of each tuple
        def get_score(tup):
            return tup[0]

        # Values for recursive call
        ret_action = None
        low_score = float('inf')
        next_beta = beta
        next_agent = self.get_next_agent(game_state, agent_index)
        next_depth = self.get_next_depth(game_state, agent_index, depth)

        for (successor, action) in self.get_succeeded_actions(game_state, agent_index):

            # Update low score and the associated action
            low_score, ret_action = min([
                (low_score, ret_action),
                (self.alpha_beta(successor, next_agent, next_depth, alpha, next_beta)[0], action)
            ], key=get_score)

            # Update the beta
            next_beta = min(next_beta, low_score)

            # No point in looking past here
            if next_beta <= alpha:
                break

        return (low_score, ret_action)

    def alpha_beta(self, game_state, agent_index, depth, alpha, beta):
        """
        Return pair of form (score, action), where:
            action = some legal move doable by agent_index from game_state
            score = score obtained by agent_index executing action
        """

        # Return if the node is terminal
        if self.is_terminal(game_state, depth):
            return (self.evaluationFunction(game_state), None)

        if self.is_pacman(agent_index):
            return self.alpha_pruned(game_state, agent_index, depth, alpha, beta)

        return self.beta_pruned(game_state, agent_index, depth, alpha, beta)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.alpha_beta(gameState, 0, self.depth, -float('inf'), float('inf'))
        return action


class ExpectimaxAgent(AssistedMultiAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, game_state, agent_index, depth):
        # Return if the node is terminal
        if self.is_terminal(game_state, depth):
            return (self.evaluationFunction(game_state), None)

        # For recursive call
        next_agent = self.get_next_agent(game_state, agent_index)
        next_depth = self.get_next_depth(game_state, agent_index, depth)

        # Iterator of (minimax score, action) pairs
        scored_actions = (
            (self.expectimax(successor, next_agent, next_depth)[0], action)
            for (successor, action)
            in self.get_succeeded_actions(game_state, agent_index)
        )

        # Score is the first element of each tuple
        def get_score(tup):
            return tup[0]

        # PacMan wants to pick the best action
        if self.is_pacman(agent_index):
            return max(scored_actions, key=get_score)

        # Ghost scores are amortized using uniform probability distribution
        else:
            scored_action_lst = list(scored_actions)
            total_score = sum((get_score(tup) for tup in scored_action_lst))
            expected = total_score / len(scored_action_lst)

            # Action taken does not matter as PacMan only looks at the score, return None
            return (expected, None)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = self.expectimax(gameState, 0, self.depth)
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Create a distance calculation function
    map_distance = MapDistance.get_calculator(currentGameState.getWalls())

    # Extracting various features
    food_dist = nearest_food_distance(currentGameState, map_distance)
    ghost_dist = total_worrisome_distance(currentGameState, 2, map_distance)
    score = currentGameState.getScore()

    # Weighing each feature
    weighted = (
        5 / food_dist,
        10 * ghost_dist,
        score
    )

    # Return the weighted sum of each feature
    return sum(weighted)


# Abbreviation
better = betterEvaluationFunction
