from griddly.util.rllib.environment.core import RLlibEnv
from watts.utils.box2d.biped_walker_custom import BipedalWalkerCustom, DEFAULT_ENV
from watts.utils.box2d.walker_wrapper import OverrideWalker
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


class WalkerFactory:
    def __init__(self, env_name: str, env_wrappers: list):
        self.env_wrappers = env_wrappers
        register_env(env_name, self.make())

    def make(self):
        def _make(env_config: dict = dict()) -> OverrideWalker:
            # The BipedalWalkerCustom env behaves like the standard BipedalWalker env
            #  however, you can pass in custom arguments that shape the terrain e.g. stump_height
            #  A future direction could also be changing the physics of the world;
            #    That's akin to changing yaml files for griddly.
            env = BipedalWalkerCustom(env_config.get('config_tuple', DEFAULT_ENV))
            # The OverrideWalker wrapper allows for consistent interaction between
            # this custom BPW env and the Griddly::RLlibEnvs
            #  e.g. env.reset(level_string='...')
            #    or env.enable_history(True), etc.
            env = OverrideWalker(env)
            return env
        return _make
