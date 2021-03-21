import os
import sys
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


if __name__ == "__main__":

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=0)

    args = load_from_yaml(fpath=_args.args_file)

    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registrar=registry, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry)

    manager = PoetManager(exp_name=_args.exp_name,
                          gym_factory=gym_factory,
                          network_factory=network_factory,
                          registrar=registry)

    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = StaticGenerator(level_string)

    manager.add_pair(network=network_factory.make()(), generator=generator)
    manager.add_pair(network=network_factory.make()(), generator=generator)
    manager.add_pair(network=network_factory.make()(), generator=generator)

    eval_futures = manager.evaluate()
    eval_returns = ray.get(eval_futures)
    for e in eval_returns:
        for k, v in e.items():
            if k == 'score':
                print(f"score for env : {sum(v)}")

    opt_futures = manager.optimize()
    opt_returns = ray.get(opt_futures)
    for e in opt_returns:
        for k, v in e.items():
            print(k, v)

    print("testing transfer")

    nets = [p.solver for p in manager.pairs]
    lvls = [p.generator for p in manager.pairs]
    new_weights = manager.transfer(nets, lvls)
    print(new_weights)

    for i, new_weight in new_weights.items():
        manager.set_solver_weights(i, new_weight)

    ray.shutdown()
