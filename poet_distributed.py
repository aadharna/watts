import argparse
import os
import ray
import sys
import time

from evolution.evolution_strategy import BirthThenKillStrategy
from evolution.replacement_strategy import ReplaceOldest
from evolution.selection_strategy import SelectRandomly
from generators.AIIDE_generator import EvolutionaryGenerator
from generators.static_generator import StaticGenerator
from gym_factory import GridGameFactory
from managers.POETManager import PoetManager
from network_factory import NetworkFactory
from pair.agent_environment_pair import Pairing
from serializer.POETManagerSerializer import POETManagerSerializer
from solvers.SingleAgentSolver import SingleAgentSolver
from transfer.score_strategy import ZeroShotCartesian
from transfer.rank_strategy import GetBestSolver
from utils.gym_wrappers import add_wrappers
from utils.register import Registrar
from utils.loader import load_from_yaml
from validators.graph_validator import GraphValidator


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, help='exp name')
parser.add_argument("--args_file", type=str, default='args.yaml', help='path to args file')
_args = parser.parse_args()


if __name__ == "__main__":

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1,
             ignore_reinit_error=True,)
             # log_to_driver=False,
             # local_mode=True)

    start = time.time()

    args = load_from_yaml(fpath=_args.args_file)

    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.get_nn_build_info)

    # not stable for non-zelda environments
    #generator = EvolutionaryGenerator(args.initial_level_string,
    #                                  file_args=registry.get_generator_config)
    generator = StaticGenerator(args.initial_level_string)

    if args.use_snapshot:
        manager = POETManagerSerializer.deserialize()
    else:
        manager = PoetManager(exp_name=_args.exp_name,
                              gym_factory=gym_factory,
                              network_factory=network_factory,
                              initial_pair=Pairing(solver=SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                                                                   trainer_config=registry.get_trainer_config,
                                                                                   registered_gym_name=registry.env_name,
                                                                                   network_factory=network_factory,
                                                                                   gym_factory=gym_factory,
                                                                                   log_id=f"{_args.exp_name}_{0}"),
                                                   generator=generator),
                              evolution_strategy=BirthThenKillStrategy(level_validator=GraphValidator(),
                                                                       replacement_strategy=ReplaceOldest(args.max_envs),
                                                                       selection_strategy=SelectRandomly(args.max_children),
                                                                       mutation_rate=args.mutation_rate),
                              transfer_strategy=GetBestSolver(ZeroShotCartesian(config=registry.get_config_to_build_rllib_env)),
                              registrar=registry)

        #import pdb; pdb.set_trace()
    try:
        manager.run()
        print("finished algorithm")
    except (Exception, KeyboardInterrupt) as e:
        error = e
        print('_'*40)
        print(error)
        print('_'*40)
    finally:
        elapsed = time.time() - start
        print(elapsed // 60, " minutes")
        # print(f"{len(manager.active_population)} PAIR objects: \n {manager.active_population}")
        ray.shutdown()
