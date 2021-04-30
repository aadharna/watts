import abc

class BaseGenerator(abc.ABC):
    def __init__(self):
        super(BaseGenerator, self).__init__()
        pass

    def mutate(self, **kwargs):
        raise NotImplementedError

    def update_from_lvl_string(self, level_string):
        raise NotImplementedError

    def generate_fn_wrapper(self):
        def _generate() -> str:
            raise NotImplementedError
        return _generate

    def __str__(self):
        """This function is necessary for our generators. This function will take whatever
        the internal representation of our "levels" are and turn it into a strong that we
        can then use to set the simulator to in Griddly.

        :return:
        """
        raise NotImplementedError
