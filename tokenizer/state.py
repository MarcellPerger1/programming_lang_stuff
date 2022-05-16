from enum import Enum, auto


class State(Enum):
    NONE = auto()

    @classmethod
    def is_none(cls, state):
        return state is None or state == cls.NONE
