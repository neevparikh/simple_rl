# Python imports.
from collections import defaultdict
from random import choice, seed
seed(10)

# Other imports.
from simple_rl.pomdp.POMDPClass import POMDP
from simple_rl.tasks.monkey_playroom.MonkeyPlayroomObjectClass import Robot, \
        Light, Ball, Monkey, Box, MusicPlayer, Bell


class MonkeyPlayroomState():
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def __hash__(self):
        return hash(tuple([hash(obj) for _, obj in self.state_dict.items()]))


class MonkeyPlayroomPOMDP(POMDP):
    ''' Class for a Monkey Playroom POMDP '''
    def __init__(self, num_rooms, cur_state=None):
        # format: for each in set of objects, maintain a list of instances
        # each having attributes

        if not cur_state:
            self.cur_state = MonkeyPlayroomState({
                'light0':
                Light('light0', False, 0),
                'light1':
                Light('light1', False, 1),
                'light2':
                Light('light2', False, 2),
                'light3':
                Light('light3', False, 3),
                'musicplayer':
                MusicPlayer('musicplayer', False, 1),
                'bell':
                Bell('bell', False, 2),
                'monkey':
                Monkey('monkey', False, 3),
                'robot':
                Robot('robot', True, 0),
                'ball':
                Ball('ball', False, 0),
                'box0':
                Box('box0', False, 0),
                'box1':
                Box('box1', False, 0),
                'box2':
                Box('box2', False, 0)
            })
        else:
            self.cur_state = cur_state

        self.start_consequence = [
            ('light0', 'toggled', False), ('light1', 'toggled', False),
            ('light2', 'toggled', False), ('light3', 'toggled', False),
            ('musicplayer', 'toggled', False), ('monkey', 'toggled', False),
            ('robot', 'room', 0),
            ('ball', 'object_in_contact',
             self.cur_state.state_dict['ball'].object_in_contact),
            ('box0', 'toggled', False), ('box1', 'toggled', False),
            ('box2', 'toggled', False)
        ]

        self.end_preconditions = [('light3', 'toggled', False),
                                  ('musicplayer', 'toggled', True),
                                  ('robot', 'room', 2),
                                  ('ball', 'object_in_contact', 'bell')]

        self.num_rooms = num_rooms

        print("Getting actions")
        self.ACTIONS = [
            obj._get_actions() for obj in self.cur_state.state_dict.values()
        ]  # pylint: disable=protected-access
        self.ACTIONS = list(
            set([action for actions in self.ACTIONS for action in actions]))

        self.ACTIONS.remove('robot_move_to_room')
        self.ACTIONS.extend(
            ['robot_move_to_room_' + str(room) for room in range(num_rooms)])

        print("Getting observations")
        self.OBSERVATIONS = ["none"]

        for room in range(num_rooms):
            in_room = []
            for k, v in self.cur_state.state_dict.items():
                if k != 'robot' and v.room == room:
                    in_room.append(k)
            self.OBSERVATIONS.append("_".join(in_room))

        tmp_obj = self.cur_state.state_dict

        print("Getting states")
        self.states = []

        for k, v in tmp_obj.items():
            if k == 'robot':
                for room in range(num_rooms):
                    v.room = room
                    tmp_obj[k] = v
                    self.states.append(MonkeyPlayroomState(tmp_obj))
            if k == 'ball':
                for k2, _ in tmp_obj.items():
                    if k2 != k:
                        v.object_in_contact = k2
                        tmp_obj[k] = v
                        self.states.append(MonkeyPlayroomState(tmp_obj))
            for tog in [0, 1]:
                if (k != 'ball' or k != 'robot'):
                    v.toggled = tog
                    tmp_obj[k] = v
                    self.states.append(MonkeyPlayroomState(tmp_obj))

        print("Initializing beliefs")
        # Initial belief is a uniform distribution over states
        num_states = len(self.states)
        print("Num states, actions, observations:", num_states, len(self.ACTIONS), len(self.OBSERVATIONS))
        b0 = defaultdict(lambda x: 1 / num_states)
        for state in self.states:
            b0[state] = 1 / num_states

        print("Initializing POMDP")
        POMDP.__init__(self, self.ACTIONS, self.OBSERVATIONS,
                       self._transition_func, self._reward_func,
                       self._observation_func, b0)

    def _transition_func(self, state, action):
        '''
        Args:
            state (dict)
            action (str)

        Returns:
            next_state (dict)
        '''
        state = MonkeyPlayroomState(state.state_dict)
        split = action.split("_")
        assert (len(split) >= 2)
        if "_".join(split[1:-1]) == "move_to_room":
            assert (int(split[-1]) <= self.num_rooms)
            getattr(state.state_dict[split[0]],
                    "_".join(split[1:-1]))(int(split[-1]))
        else:
            getattr(state.state_dict[split[0]], "_".join(split[1:]))()
        if action == 'ball_throw' and state.state_dict[
                'ball'].object_in_contact == 'robot':
            state['ball'].object_in_contact = choice(
                state['ball'].throwable[str(state['robot'].room)])
        if self.is_goal_state_action(state, action):
            state['monkey'].toggled = 1
        return state

    def _observation_func(self, state, action):
        next_state = self._transition_func(state, action).state_dict
        cur_room = next_state['robot'].room
        if next_state['light' + str(cur_room)]:
            return self.OBSERVATIONS[cur_room + 1]
        else:
            return self.OBSERVATIONS[0]

    def _reward_func(self, state, action, next_state):  # pylint: disable=unused-argument
        if self.is_goal_state_action(state, action):
            return 1
        else:
            return -0.01

    def get_precondition(self, action):
        split = action.split("_")
        assert (len(split) >= 2)
        return self.cur_state.state_dict[split[0]].preconditions['_'.join(
            split[1:])]

    def get_consequence(self, action):
        split = action.split("_")
        assert (len(split) >= 2)
        return self.cur_state.state_dict[split[0]].consequences['_'.join(
            split[1:])]

    def is_goal_state_action(self, state, action):
        state = state.state_dict
        return (action == 'ball_throw' and not (state['light3'].toggled)
                and state['musicplayer']
                and state['ball'].object_in_contact == 'bell'
                and state['robot'].room == 2)

    def get_all_agent_states(self):
        return self.states

    def is_in_goal_state(self):
        return self.cur_state.state_dict['monkey'].toggled


if __name__ == '__main__':
    monkey_playroom_pomdp = MonkeyPlayroomPOMDP(4)
