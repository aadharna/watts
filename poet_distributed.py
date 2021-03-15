import ray

import time
import argparse

from POETManager import PoetManager
from gym_factory import GridGameFactory
from network_factory import NetworkFactory

from generators.static_generator import StaticGenerator

from utils.gym_wrappers import add_wrappers
from utils.register import register_env_with_rllib
from utils.loader import load_from_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, help='exp name')
parser.add_argument("--args_file", type=str, default='args.yaml', help='path to args file')
_args = parser.parse_args()


@ray.remote
def f(i):
    time.sleep(1)
    return i


if __name__ == "__main__":

    ray.init()

    args = load_from_yaml(fpath=_args.args_file)

    name, nActions, actSpace, obsSpace, observer = register_env_with_rllib(file_args=args)

    wrappers = add_wrappers(args.wrappers)

    gym_factory = GridGameFactory(file_args=args, name=name, n_actions=nActions, act_space=actSpace, obs_space=obsSpace,
                                  observer=observer, env_wrappers=wrappers)

    network_factory = NetworkFactory(obs_space=obsSpace,
                                     action_space=actSpace,
                                     num_outputs=nActions,
                                     model_config={},
                                     name=args.network_name)

    manager = PoetManager(_args.exp_name, file_args=args, gym_factory=gym_factory, network_factory=network_factory)
    """
    # Paired Open Ended Trailblazer main loop
    #
    # For forever:
    #
    # 1) Build new environments (or change robot morphology)
    #    This could take the form of:
    #       * evolutionary algorithm
    #       * neural network
    #       * random domains
    #       * etc
    #
    #    1b) Make sure new environments are not too easy or too hard
    #    1c) When too many envs, remove envs that are too old (or some other metric)

    # 2) Optimize agents in their current environment
    #    This could be done with any sort of optimization algorithm
    #       * Evolutionary methods
    #       * RL
    #       * etc

    # 3) Transfer agents across environments
    #    This could be determined with
    #       * self-play evaluations
    #       * tournament selection
    #       * etc

    """
    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = StaticGenerator(level_string)

    manager.add_pair(network=network_factory.make()(), generator=generator)
    manager.add_pair(network=network_factory.make()(), generator=generator)

    eval_futures = manager.evaluate()
    eval_returns = ray.get(eval_futures)
    for e in eval_returns:
        for k, v in e.items():
            if k == 'score':
                print(f"score for env : {sum(v)}")

    ray.shutdown()
