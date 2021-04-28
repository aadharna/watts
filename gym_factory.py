from griddly.util.rllib.environment.core import RLlibEnv
from ray.tune.registry import register_env


class GridGameFactory:
    def __init__(self, registrar, env_wrappers: list):
        """Factory to create new gym envs and register it with ray's global env register

        :param registrar: utils.registry.Registrar class. This holds info needed to make new envs
        :param env_wrappers: list of env.Wrappers to apply to the env
        """

        self.registrar = registrar
        self.args = self.registrar.file_args
        self.env_wrappers = env_wrappers
        register_env(self.registrar.name, self.make())

    def make(self):
        def _make(env_config: dict = dict()) -> RLlibEnv:  # this RLlibEnv is (potentially) wrapped.
            """function used to register env creation with rllib.

            :param env_config: unused param. Here for compatibility purposes with rllib
            :return: (wrapped) RLlibEnv from griddly
            """
            env = RLlibEnv(env_config)
            env.enable_history(True)
            for i, wrapper in enumerate(self.env_wrappers):
                env = wrapper(env, env_config)
            return env
        return _make


if __name__ == "__main__":
    from utils.register import Registrar
    from utils.gym_wrappers import AlignedReward
    from utils.loader import load_from_yaml
    import os

    args = load_from_yaml(os.path.join('args.yaml'))

    registry = Registrar(file_args=args)

    gameFactory = GridGameFactory(registrar=registry, env_wrappers=[AlignedReward])

    env = gameFactory.make()()#registry.get_config_to_build_rllib_env)
    import matplotlib.pyplot as plt

    state = env.reset()
    plt.imshow(state.swapaxes(0, 2))
    plt.show()
