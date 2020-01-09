# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.tasks.monkey_playroom.MonkeyPlayroomStateClass import MonkeyPlayroomState
from simple_rl.tasks.monkey_playroom.MonkeyPlayroomObjectClass import Robot, Light, Ball, Monkey, Box, MusicPlayer, Bell


class MonkeyPlayroomPOMDP(POMDP):
    ''' Class for a Monkey Playroom POMDP '''
    def __init__(self, num_rooms):
        # format: for each in set of objects, maintain a list of instances
        # each having attributes

        self.OBSERVATIONS = ['']

        self.cur_state = {
            'light0': Light('light0', False, 0),
            'light1': Light('light1', False, 1),
            'light2': Light('light2', False, 2),
            'light3': Light('light3', False, 3),
            'music_player': MusicPlayer('music_player', False, 1),
            'bell': Bell('bell', False, 2),
            'monkey': Monkey('monkey', False, 3),
            'robot': Robot('robot', True, 0),
            'ball': Ball('ball', False, 0),
            'box0': Box('box0', False, 0),
            'box1': Box('box1', False, 0),
            'box2': Box('box2', False, 0)
        }

        self.ACTIONS = [obj._get_actions() for obj in self.cur_state.values()]  # pylint: disable=protected-access
        self.ACTIONS = list(
            set([action for actions in self.ACTIONS for action in actions]))

        self.num_rooms = num_rooms
        tmp_obj = self.cur_state
        self._states = []
        for k, v in tmp_obj.items():
            for tog in [0, 1]:
                v.toggled = tog
                tmp_obj[k] = v
                self._states.append(MonkeyPlayroomState(tmp_obj))
            if k == 'robot':
                for room in range(num_rooms):
                    v.room = room
                    tmp_obj[k] = v
                    self._states.append(MonkeyPlayroomState(tmp_obj))
            if k == 'ball':
                for k2, _ in tmp_obj.items():
                    if k2 != k:
                        v.object_in_contact = k2
                        tmp_obj[k] = v
                        self._states.append(MonkeyPlayroomState(tmp_obj))

        # Initial belief is a uniform distribution over states
        b0 = defaultdict()
        for state in self._states:
            b0[state] = 1 / len(self._states)

        POMDP.__init__(self, self.ACTIONS, self.OBSERVATIONS,
                       self._transition_func, self._reward_func,
                       self._observation_func, b0)

    def _transition_func(self, state, action):
        '''
        Args:
            state (MonkeyPlayroomState)
            action (str)

        Returns:
            next_state (MonkeyPlayroomState)
        '''

        raise ValueError(
            'Invalid state: {} action: {} in MonkeyPlayroom'.format(
                state, action))

    def _observation_func(self, state, action):
        pass

    def _reward_func(self, state, action, next_state):  # pylint: disable=unused-argument
        if next_state['monkey'].toggled:
            return 1
        else:
            return -0.05

    def get_all_agent_states(self):
        return self._states

    def is_in_goal_state(self):
        return self.cur_state['monkey'].toggled


if __name__ == '__main__':
    monkey_playroom_pomdp = MonkeyPlayroomPOMDP(4)
