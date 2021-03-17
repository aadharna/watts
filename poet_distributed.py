import ray

import time
import argparse

from managers.POETManager import PoetManager
from gym_factory import GridGameFactory
from network_factory import NetworkFactory

from generators.static_generator import StaticGenerator

from utils.gym_wrappers import add_wrappers
from utils.register import Registrar
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

    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(file_args=args, env_wrappers=wrappers, registrar=registry)
    network_factory = NetworkFactory(registry)

    manager = PoetManager(_args.exp_name, file_args=args, gym_factory=gym_factory, network_factory=network_factory)

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
