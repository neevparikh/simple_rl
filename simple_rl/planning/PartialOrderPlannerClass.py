from simple_rl.planning import Planner
from simple_rl.pomdp.POMDPClass import POMDP
from sortedcontainers import SortedList


class POP_Bubble:
    """ Bubble class for POP Planner """
    def __init__(self, pop_actions, action_position):
        self.actions = set(pop_actions)
        self.position = action_position

    def __lt__(self, other):
        return self.position < other.position

class POP_Action:
    """ Action class for POP Planner """
    def __init__(self, action, bubble_idx, preconditions, consequences):
        self.action = action
        self.bubble_idx = bubble_idx
        self.preconditions = set(preconditions)
        self.consequences = set(consequences)

class PartialOrderPlanner(Planner):
    def __init__(self, pomdp, name='POP'):

        Planner.__init__(self, super(POMDP, pomdp), name)

        # Defining sentinal actions
        # start_action = POP_Action(["__start__"], 0)
        # end_action = POP_Action(["__end__"], float("inf"))
        self.ordering = SortedList([start_action, end_action])
        self.agenda = set()
        self.delayed_agenda = set()

    def _is_threat_(self, action, link):
        return False

    def choose_subgoal(self, agenda):
        pass

    def plan(self):
        pass

    def policy(self):
        pass
