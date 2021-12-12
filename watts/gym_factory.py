from griddly.util.rllib.environment.core import RLlibEnv
from ray.tune.registry import register_env


class GridGameFactory:
    def __init__(self, env_name: str, env_wrappers: list):
        """Factory to create new gym envs and register it with ray's global env register

        :param env_name: string name of the game that we want to register with RLlib
        :param env_wrappers: list of env.Wrappers to apply to the env
        """

        self.env_wrappers = env_wrappers
        register_env(env_name, self.make())

    def make(self):
        def _make(env_config: dict = dict()) -> RLlibEnv:  # this RLlibEnv is (potentially) wrapped.
            """function used to register env creation with rllib.

            :param env_config: information about how to build the env; compatible with Griddly and RLlibEnv
            :return: (wrapped) RLlibEnv from griddly
            """
            env = RLlibEnv(env_config)
            env.enable_history(True)
            for wrapper in self.env_wrappers:
                env = wrapper(env, env_config)
            return env
        return _make
