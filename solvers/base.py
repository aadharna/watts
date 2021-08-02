class BaseSolver:
    id = 0

    def __init__(self):
        self.id = BaseSolver.id
        BaseSolver.id += 1

    def evaluate(self, env_generator_fn, env_config) -> dict:
        raise NotImplementedError

    def optimize(self, trainer_config, level_string_monad, **kwargs):
        raise NotImplementedError

    def get_weights(self) -> list:
        raise NotImplementedError

    def set_weights(self, new_weights):
        raise NotImplementedError
