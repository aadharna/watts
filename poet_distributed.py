import argparse
import os
import ray
import sys

from gym_factory import GridGameFactory
from managers.POETManager import PoetManager
from mutation.mutation_strategy import EvolveStrategy
from transfer.score_strategy import ZeroShotCartesian
from transfer.rank_strategy import GetBestSolver
from mutation.level_validator import GraphValidator, RandomVariableValidator
from mutation.level_validator import PINSKYValidator
from network_factory import NetworkFactory
from generators.static_generator import StaticGenerator
from generators.AIIDE_generator import EvolutionaryGenerator
from pair.agent_environment_pair import Pairing

from solvers.SingleAgentSolver import SingleAgentSolver

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

    ray.init(num_gpus=1, ignore_reinit_error=True)#, local_mode=True)

    args = load_from_yaml(fpath=_args.args_file)

    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info)

    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = EvolutionaryGenerator(level_string, file_args=registry.get_generator_config)

    manager = PoetManager(exp_name=_args.exp_name,
                          gym_factory=gym_factory,
                          network_factory=network_factory,
                          initial_pair=Pairing(solver=SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                                                               trainer_config=registry.get_trainer_config,
                                                                               registered_gym_name=registry.env_name,
                                                                               network_factory=network_factory,
                                                                               gym_factory=gym_factory),
                                               generator=generator),
                          mutation_strategy=EvolveStrategy(PINSKYValidator(gym_factory_monad=gym_factory.make(),
                                                                           env_config=registry.get_config_to_build_rllib_env),
                                                           args.max_children,
                                                           args.mutation_rate),
                          transfer_strategy=GetBestSolver(ZeroShotCartesian(gym_factory=gym_factory,
                                                                            config=registry.get_config_to_build_rllib_env)),
                          registrar=registry)

    try:
        manager.run()
        print("finished algorithm")
    except Exception as e:
        error = e
        print(error)
        print('_'*40)

    print(f"{len(manager.pairs)} PAIR objects: \n {manager.pairs}")
    ray.shutdown()
