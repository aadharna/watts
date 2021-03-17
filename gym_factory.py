import gym
from griddly import gd
from utils.loader import load_from_yaml

class GridGameFactory:
    def __init__(self, file_args, env_wrappers: list, registrar):
        """Factory to create new gym envs.

        :param registrar: utils.registry.Registrar class. This holds info needed to make new envs
        :param file_args: arguments loaded from file via utils.loader.load_yaml_file
        :param env_wrappers: list of env.Wrappers to apply to the env
        """

        self.registrar = registrar
        self.args = file_args
        self.env_wrappers = env_wrappers

    def make(self):
        def _make():
            from ray.tune.registry import register_env
            from griddly.util.rllib.wrappers.core import RLlibEnv
            register_env(self.registrar.env_name, RLlibEnv)
            env = RLlibEnv(self.registrar.get_config_to_build_rllib_env)
            env.enable_history(True)
            for wrapper in self.env_wrappers:
                env = wrapper(env)
                if 'aligned' in str(env) and env.play_length is None:
                    env.play_length = self.args.game_len
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

    env = gameFactory.make()()
    import matplotlib.pyplot as plt

    state = env.reset()
    plt.imshow(state.swapaxes(0, 2))
    plt.show()
