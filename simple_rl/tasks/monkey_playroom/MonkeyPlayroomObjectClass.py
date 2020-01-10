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

    def _get_actions(self):
        # return actions
        return [
             self.name + '_' + method for method in dir(self)
            if method[0] != '_'
        ]


class Robot(MonkeyPlayroomObject):
    def move_to_room(self, room):
        self.room = room


class Monkey(MonkeyPlayroomObject):
    pass


class Box(MonkeyPlayroomObject):
    def open(self):
        self.toggled = True

    def close(self):
        self.toggled = False


class Ball(MonkeyPlayroomObject):
    def __init__(self, object_name, object_state, room):
        self.object_in_contact = 'box' + str(randint(0, 2))
        super(Ball, self).__init__(object_name, object_state, room)

    def throw(self):
        self.toggled = True


class MusicPlayer(MonkeyPlayroomObject):
    def turn_on(self):
        self.toggled = True

    def turn_off(self):
        self.toggled = False


class Light(MonkeyPlayroomObject):
    def turn_on(self):
        self.toggled = True

    def turn_off(self):
        self.toggled = False


class Bell(MonkeyPlayroomObject):
    def turn_on(self):
        self.toggled = True

    def turn_off(self):
        self.toggled = False
