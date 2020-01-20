from simple_rl.planning import Planner
from random import seed
from collections import defaultdict
seed(10)


class POP_Action:
    """ Action class for POP Planner """
    def __init__(self, action, preconditions, consequences):
        self.action_name = action
        self.position = None
        self.preconditions = set([(p, action) for p in preconditions])
        self.consequences = set([(c, action) for c in consequences])

    def __lt__(self, other):
        return self.position < other.position


class PartialOrderPlanner(Planner):
    def __init__(self, pomdp, name='POP'):

        Planner.__init__(self, pomdp, name)

        self.pomdp = pomdp
        self.remaining_actions = set([
            POP_Action(action, pomdp.get_precondition(action),
                       pomdp.get_consequence(action))
            for action in pomdp.ACTIONS
        ])

        # Defining sentinal actions
        self.start_action = POP_Action("__start__", [],
                                       pomdp.start_consequence)
        self.end_action = POP_Action("__end__", pomdp.end_preconditions, [])

        self.ordering = defaultdict(list)
        assert (not self.ordering[self.end_action])
        self._add_ordering(self.start_action, self.end_action)
        self.planned_actions = set([self.start_action, self.end_action])
        self.agenda = set().union(self.end_action.preconditions)
        self.delayed_agenda = set()
        self.links = {}

    def _check_consistency(self, new_action, prev_action):
        return not new_action.action_name in self.ordering[
            prev_action.action_name]

    def _add_ordering(self, action_1, action_2):
        assert (action_2 in self.ordering.keys())
        self.ordering[action_1.action_name].append(action_2.action_name)
        self.ordering[action_1.action_name].extend(
            self.ordering[action_2.action_name])

    def _protect_link(self, action, action_1, Q, action_2):
        if self._check_consistency(action, action_1):
            self._add_ordering(action, action_1)
        elif self._check_consistency(action_2, action):
            self._add_ordering(action_2, action)
        else:
            __import__('pdb').set_trace()
            raise RuntimeError(
                "Could not resolve threat with: {}, {}, {} and {}".format(
                    action, action_1, action_2, Q))

    def plan(self):
        while self.agenda:
            Q, A_need = self.agenda.pop()

            # get A_add from new actions:
            possible_actions = set(
                filter(lambda act: (Q, act) in act.preconditions,
                       self.remaining_actions))
            if possible_actions:
                A_add = possible_actions.pop()
                self.remaining_actions.remove(A_add)
                self.links[Q] = (A_add, A_need)
                self._add_ordering(A_add, self.end_action)
                self._add_ordering(self.start_action, A_add)
                self.planned_actions.add(A_add)
                self.agenda.update(A_add.preconditions)
            else:
                # pick A_add from planned actions?
                possible_actions = set(
                    filter(lambda act: (Q, act) in act.preconditions,
                           self.planned_actions))
                if possible_actions and self._check_consistency(A_add, A_need):
                    A_add = possible_actions.pop()
                    self.links[Q] = (A_add, A_need)
                    self._add_ordering(A_add, A_need)
                else:
                    __import__('pdb').set_trace()
                    raise RuntimeError("A_add could not be picked.")

            # Do causal link protection
            for R, actions in self.links.items():
                self._protect_link(A_add, actions[0], R, actions[1])

            for action in self.planned_actions:
                self._protect_link(action, A_add, Q, A_need)

        return self.policy()

    def policy(self):
        # convert ordering into list of actions or mapping s -> a?
        print(self.ordering[self.start_action.action_name])
