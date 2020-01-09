from simple_rl.mdp.StateClass import State


class MonkeyPlayroomState(State):
    ''' Class for MonkeyPlayroom POMDP States '''
    def __init__(self, state_dict):
        data = state_dict
        if 'monkey' not in state_dict.keys():
            raise ValueError('State dict must have monkey as key')
        is_terminal = state_dict['monkey'].toggled 
        State.__init__(self, data=data, is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return 'MonkeyPlayroomState::{}'.format(self.data)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other,
                          MonkeyPlayroomState) and self.data == other.data
