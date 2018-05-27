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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return DfsProcedure(problem).run()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return BfsProcedure(problem).run()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return UcsProcedure(problem).run()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return AStarProcedure(problem, heuristic).run()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


# region Search Algorithm Implementation

# TODO DOCSTRINGS AND REFACTOR FOR FASTER COST COMPUTATION

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


# Partial implementation of SearchProcedure
# that uses path checking in get_successors
# (states of each path must be unique WITHIN THEMSELVES)
class PathCheckedProcedure(SearchProcedure):
    def get_successors(self, path):

        # Filter down successors by those whose states are not in the path
        ret = []
        for successor in super(PathCheckedProcedure, self).get_successors(path):
            state = self.get_state_of_successor(successor)
            if not path.has_state(state):
                ret += [successor]

        return ret


# Complete implementation of DFS as a PathCheckedProcedure
class DfsProcedure(PathCheckedProcedure):
    def __init__(self, problem):
        super(DfsProcedure, self).__init__(problem)

    def init_fringe(self):
        self.fringe = util.Stack()

    def fringe_is_empty(self):
        return self.fringe.isEmpty()

    def push_path(self, path):
        self.fringe.push(path)

    def pop_path(self):
        return self.fringe.pop()

    def get_edge_of_successor(self, succ):
        return succ[1]

    def get_action_of_edge(cls, edge):
        return edge


# Partial implementation of SearchProblemProcedure
# that uses cycle checking in get_successors
# (states of each path must be unique ACROSS ALL PATHS)
class CycleCheckedProcedure(SearchProcedure):
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


# Complete implementation of BFS as a CycleCheckedProcedure
class BfsProcedure(CycleCheckedProcedure):
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


# Wrapper for SearchPaths
# Instances are considered equal if they have the same state -
# this is so PriorityQueue can be updated appropriately
# TODO REFACTOR CODE TO CACHE COSTS
class StateEquatablePath(object):
    def __init__(self, search_path):
        self.path = search_path

    def __eq__(self, other):
        return self.path.state == other.path.state


# Partial implementation of CycleCheckedProcedure that uses a
# priority queue fringe and cycle checking
class PriorityQueueProcedure(CycleCheckedProcedure):
    def queue_priority(self, path):
        util.raiseNotDefined()

    def init_fringe(self):
        self.fringe = util.PriorityQueue()

    def fringe_is_empty(self):
        return self.fringe.isEmpty()

    def push_path_actual(self, path):

        # Figure out the priority, wrap the path so that queue updates work
        priority = self.queue_priority(path)
        eq_path = StateEquatablePath(path)

        # Push if we haven't opened the path yet (faster performance of O(logN))
        if path.state not in self.open_states:
            self.fringe.push(eq_path, priority)

        # Otherwise update the priority (slower performance of O(n))
        else:
            self.fringe.update(eq_path, priority)

    def pop_path_actual(self):
        eq_path = self.fringe.pop()
        return eq_path.path

    # Must consider opened paths
    def get_successors(self, path):
        return super(PriorityQueueProcedure, self).get_successors(path, True)

    # Edges are going to be doubles of form (action, cost)
    def get_edge_of_successor(self, succ):
        return succ[1:]

    # Action is the first component of an edge
    def get_action_of_edge(cls, edge):
        return edge[0]


# Complete implementation of UCS as a PriorityQueueProcedure
class UcsProcedure(PriorityQueueProcedure):
    def queue_priority(self, path):
        # If the edge is None we're on the first path
        # It doesn't matter what priority we choose
        # because it will be immediately popped from the queue
        if path.edge is None:
            return 0

        # Cost is the CUMULATIVE cost of the path (cost is second component of edge)
        return sum((edge[1] for edge in path.to_edges()))


# Complete implementation of A-Star as a PriorityQueueProcedure
class AStarProcedure(PriorityQueueProcedure):
    def __init__(self, problem, heuristic):
        super(AStarProcedure, self).__init__(problem)
        self.heuristic = heuristic

    def queue_priority(self, path):
        # If the edge is None we're on the first path
        # It doesn't matter what priority we choose
        # because it will be immediately popped from the queue
        if path.edge is None:
            return 0

        # Cost is the heuristic + cumulative edge cost
        return self.heuristic(path.state, self.problem) + \
               sum((edge[1] for edge in path.to_edges()))

# endregion
