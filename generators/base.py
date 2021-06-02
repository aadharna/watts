import abc
from typing import Tuple


class BaseGenerator(abc.ABC):
    def __init__(self):
        super(BaseGenerator, self).__init__()
        pass

    def mutate(self, **kwargs):
        raise NotImplementedError

    def update(self, level):
        raise NotImplementedError

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            raise NotImplementedError
        return _generate

    def __str__(self):
        """This function is necessary for our generators. This function will take whatever
        the internal representation of our "levels" are and turn it into a string that we
        can then use to set the simulator to in Griddly.

        :return:
        """
        raise NotImplementedError
