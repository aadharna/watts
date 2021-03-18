from griddly.util.rllib.wrappers.core import RLlibEnv
from ray.tune.registry import register_env

class GridGameFactory:
    def __init__(self, file_args, env_wrappers: list, registrar):
        """Factory to create new gym envs and register it with ray's global env register

        :param registrar: utils.registry.Registrar class. This holds info needed to make new envs
        :param file_args: arguments loaded from file via utils.loader.load_yaml_file
        :param env_wrappers: list of env.Wrappers to apply to the env
        """

        self.registrar = registrar
        self.args = file_args
        self.env_wrappers = env_wrappers
        register_env(self.registrar.name, self.make())

    def make(self):
        def _make(env_config: dict = dict()):
            """function used to register env creation with rllib.

            :param env_config: unused param. Here for compatibility purposes with rllib
            :return: (wrapped) RLlibEnv from griddly
            """
            env = RLlibEnv(self.registrar.get_config_to_build_rllib_env)
            env.enable_history(True)
            for i, wrapper in enumerate(self.env_wrappers):
                env = wrapper(env, self.registrar.get_config_to_build_rllib_env)
            return env
        return _make


if __name__ == "__main__":
    from utils.register import Registrar
    from utils.gym_wrappers import AlignedReward
    from utils.loader import load_from_yaml
    import os

    args = load_from_yaml(os.path.join('args.yaml'))

    registry = Registrar(file_args=args)

    gameFactory = GridGameFactory(file_args=args, env_wrappers=[AlignedReward], registrar=registry)

    env = gameFactory.make()()#registry.get_config_to_build_rllib_env)
    import matplotlib.pyplot as plt

    state = env.reset()
    plt.imshow(state.swapaxes(0, 2))
    plt.show()
