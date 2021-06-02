class BaseSolver:
    id = 0

    def __init__(self):
        self.id = BaseSolver.id
        BaseSolver.id += 1

    @staticmethod
    def evaluate(actors, env) -> dict:
        raise NotImplementedError

    def get_weights(self) -> list:
        raise NotImplementedError

    def set_weights(self, new_weights):
        raise NotImplementedError
