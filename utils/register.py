import os
import gym
from gym.spaces import MultiDiscrete, Discrete
from numpy import prod
import griddly
from griddly import gd
from griddly.util.rllib.wrappers.core import RLlibEnv

from ray.tune.registry import register_env

from utils.loader import load_from_yaml


def register_env_with_griddly(file_args) -> tuple:
    """Register a GDY file with griddly to create the environment via gym.make()
    Make the environment to access action/observation types

    :param args_file: path to file with arguments e.g. game and folder with yamls
    :param GDY_prefix: prefix for getting to the GDY yaml file.
    :return: tuple of
                (game_name, (for `gym.make()`)
                 nActions,
                 action_space,
                 observation_space,
                 observer_type: e.g. gd.Observer.VECTOR)
    """

    args = file_args

    game = args.game
    init_level_id = args.init_lvl
    dir_path = args.lvl_dir

    if args.engine == 'GDY':
        from griddly import gd
        from griddly import GymWrapperFactory
        wrapper = GymWrapperFactory()

        if args.pictures:
            observer = gd.ObserverType.SPRITE_2D
        else:
            observer = gd.ObserverType.VECTOR
        file = os.path.join(dir_path, f'{game}.yaml')
        print(f"game file at: {file} == {os.path.exists(file)}")
        try:
            wrapper.build_gym_from_yaml(
                environment_name=f'{game}-custom',
                yaml_file=file,
                level=init_level_id,
                max_steps=args.game_len,
                global_observer_type=observer,
                player_observer_type=observer
            )
        except gym.error.Error:
            pass

    else:
        raise ValueError("gvgai is not supported anymore. Please use Griddly.")

    env = gym.make(f"{args.engine}-{game}-custom-v0")
    _ = env.reset()
    # count the number of distinct discrete actions
    # THIS ASSUMES ACTION SPACES ARE DISCRETE
    actionType = env.action_space
    obsType = env.observation_space
    if type(actionType) == MultiDiscrete:
        nActions = prod(env.action_space.nvec)
    elif type(actionType) == Discrete:
        nActions = actionType.n
    else:
        raise ValueError(f"Unsupported action type in game: {game}. "
                         f"Only Discrete and MultiDiscrete are supported")

    del env

    return f"{args.engine}-{game}-custom-v0", nActions, actionType, obsType, observer


def register_env_with_rllib(file_args) -> tuple:
    """
    Register the GDY file with griddly.
    Then register the newly created gym env with RLLib using the
      griddly build-in RLlibEnv.

    :param file_args: file with args for algo and path to GDY file + game
                        This is loaded by utils.loader.load_yaml_file.
    :return: return values from `register_env_with_griddly`
    """
    name, nActions, actSpace, obsSpace, observer = register_env_with_griddly(file_args=file_args)
    register_env(name, RLlibEnv)
    return name, nActions, actSpace, obsSpace, observer


if __name__ == "__main__":
    from utils.loader import load_from_yaml
    import os
    os.chdir('..')
    arg_path = os.path.join('args.yaml')
    file_args = load_from_yaml(arg_path)
    # print(f"args at: {arg_path} == {os.path.exists(arg_path)}")
    name, nActions, actSpace, obsSpace, observer = register_env_with_rllib(file_args=file_args)
    print(name, nActions, actSpace, obsSpace, observer)