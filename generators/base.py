import abc

class BaseGenerator(abc.ABC):
    def __init__(self):
        super(BaseGenerator, self).__init__()
        pass

    def mutate(self, **kwargs):
        raise NotImplementedError

    def update_from_lvl_string(self, level_string):
        raise NotImplementedError

    def generate(self):
        def _generate() -> str:
            raise NotImplementedError
        return _generate

    def __str__(self):
        raise NotImplementedError
