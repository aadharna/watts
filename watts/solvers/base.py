class BaseSolver:

    def __init__(self):
        pass

    def evaluate(self, env_config, solver_id, generator_id) -> dict:
        raise NotImplementedError

    def optimize(self, trainer_config, level_string_monad, **kwargs):
        raise NotImplementedError

    def get_weights(self) -> dict:
        raise NotImplementedError

    def set_weights(self, new_weights):
        raise NotImplementedError

    def get_key(self):
        raise NotImplementedError
