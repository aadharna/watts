import gym
from griddly import gd
from utils.loader import load_from_yaml

class GridGameFactory:
    def __init__(self,
                 file_args,
                 name: str,
                 nActions: int,
                 actSpace: gym.spaces.Space,
                 obsSpace: gym.spaces.Space,
                 observer,
                 env_wrappers: list):
        """Factory to create new gym envs.

        gameFactory = GridGameFactory(*register_env_with_griddly(args_file='args.yaml'))

        Alternatively with rllib:
        gameFactory = GridGameFactory(*register_with_rllib(args_file='args.yaml'))

        :param args_file: arguments loaded from file via utils.loader.load_yaml_file
        :param name: name to be used with the gym.make command. e.g. GDY-Zelda-v0
        :param nActions: number of discrete actions in the env
        :param actSpace: gym action_space
        :param obsSpace: gym observation_space
        :param env_wrappers: list of env.Wrappers to apply to the env
        """
        # super(GridGame, self).__init__()

        self.args = file_args
        self.name = name
        self.nActions = nActions
        self.action_space = actSpace
        self.observation_space = obsSpace
        self.observer = observer
        self.env_wrappers = env_wrappers

    def make(self):
        def _make():
            from utils.register import register_env_with_rllib
            _ = register_env_with_rllib(file_args=self.args)
            env = gym.make(self.name,
                           global_observer_type=self.observer,
                           player_observer_type=self.observer)
            env.enable_history(True)
            for wrapper in self.env_wrappers:
                env = wrapper(env)
                if 'aligned' in str(env):
                    env.play_length = self.args.game_len
            return env
        return _make


if __name__ == "__main__":
    from utils.register import register_env_with_rllib
    from utils.gym_wrappers import AlignedReward
    from utils.loader import load_from_yaml
    import os

    args = load_from_yaml(os.path.join('args.yaml'))

    name, nActions, actSpace, obsSpace, observer = register_env_with_rllib(file_args=args)

    gameFactory = GridGameFactory(args, name, nActions, actSpace, obsSpace, observer, [AlignedReward])

    env = gameFactory.make()()
    import matplotlib.pyplot as plt

    state = env.reset()
    plt.imshow(state.swapaxes(0, 2))
    plt.show()
