from random import randint, seed
seed(10)


class MonkeyPlayroomObject(object):
    """Class for MonkeyPlayroomObject. """
    def __init__(self, object_name, object_state, room):
        """
        Must pass in the type, objectstate, list of actions and room
        """
        self.name = object_name
        self.toggled = object_state
        self.room = room

    def __hash__(self):
        return hash((self.name, self.toggled, self.room))

    def _get_actions(self):
        # return actions
        blacklist = [
            'preconditions', 'consequences', 'name', 'toggled', 'room',
            'throwable', 'object_in_contact'
        ]
        return [
            self.name + '_' + method for method in dir(self)
            if method[0] != '_' and method not in blacklist
        ]


class Robot(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        super(Robot, self).__init__(object_name, object_state, room)
        self.preconditions = {
                'move_to_room_0': [],
                'move_to_room_1': [],
                'move_to_room_2': [],
                'move_to_room_3': []
                }
        self.consequences = {
                'move_to_room_0': [(object_name, 'room', room)],
                'move_to_room_1': [(object_name, 'room', room)],
                'move_to_room_2': [(object_name, 'room', room)],
                'move_to_room_3': [(object_name, 'room', room)]
                }

    def move_to_room(self, room):
        self.room = room


class Monkey(MonkeyPlayroomObject):
    pass


class Box(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        super(Box, self).__init__(object_name, object_state, room)
        self.preconditions = {
            'open': [('robot', 'room', room),
                     ('light' + str(room), 'toggled', True)],
            'close': [('robot', 'room', room),
                      ('light' + str(room), 'toggled', True)]
        }
        self.consequences = {
            'open': [(object_name, 'toggled', True)],
            'close': [(object_name, 'toggled', False)]
        }

    def open(self):
        self.toggled = True

    def close(self):
        self.toggled = False


class Ball(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        self.object_in_contact = 'box' + str(randint(0, 2))
        self.throwable = {
            '0': ['box0', 'box1', 'box2'],
            '1': ['music_player'],
            '2': ['bell'],
            '3': ['monkey']
        }
        super(Ball, self).__init__(object_name, object_state, room)
        self.preconditions = {
            'throw': [('ball', 'object_in_contact', 'robot')]
        }
        self.consequences = {
            'throw':
            [(object_name, 'toggled', True),
             (object_name, 'object_in_contact', self.object_in_contact)]
        }

    def throw(self):
        self.toggled = True


class MusicPlayer(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        super(MusicPlayer, self).__init__(object_name, object_state, room)
        self.preconditions = {
            'turn_on': [('robot', 'room', room),
                        ('light' + str(room), 'toggled', True)],
            'turn_off': [('robot', 'room', room),
                         ('light' + str(room), 'toggled', True)]
        }
        self.consequences = {
            'turn_on': [(object_name, 'toggled', True)],
            'turn_off': [(object_name, 'toggled', False)]
        }

    def turn_on(self):
        self.toggled = True

    def turn_off(self):
        self.toggled = False


class Bell(MonkeyPlayroomObject):
    pass


class Light(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        super(Light, self).__init__(object_name, object_state, room)
        self.preconditions = {
            'turn_on': [('robot', 'room', room)],
            'turn_off': [('robot', 'room', room)]
        }
        self.consequences = {
            'turn_on': [(object_name, 'toggled', True)],
            'turn_off': [(object_name, 'toggled', False)]
        }

    def turn_on(self):
        self.toggled = True

    def turn_off(self):
        self.toggled = False
