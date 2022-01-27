import argparse
import os
import ray
import sys
import time
import pickle
from collections import OrderedDict

from watts.evolution.evolution_strategy import BirthThenKillStrategy, POETStrategy
from watts.evolution.replacement_strategy import ReplaceOldest
from watts.evolution.selection_strategy import SelectRandomly
from watts.game.GameSchema import GameSchema
from watts.generators import EvolutionaryGenerator, StaticGenerator
from watts.generators.WalkerConfigGenerator import WalkerConfigGenerator
from watts.gym_factory import GridGameFactory, WalkerFactory
from watts.managers.POETManager import PoetManager
from watts.network_factory import NetworkFactory
from watts.pair.agent_environment_pair import Pairing
from watts.serializer.POETManagerSerializer import POETManagerSerializer
from watts.solvers.SingleAgentSolver import SingleAgentSolver
from watts.transfer.score_strategy import ZeroShotCartesian
from watts.transfer.rank_strategy import GetBestSolver, GetBestZeroOrOneShotSolver
from watts.utils.gym_wrappers import add_wrappers
from watts.utils.register import Registrar
from watts.utils.loader import load_from_yaml, save_obj
from watts.validators.agent_validator import ParentCutoffValidator
from watts.validators.level_validator import AlwaysValidator, RandomVariableValidator
from watts.validators.graph_validator import GraphValidator
from watts.validators.rank_novelty_validator import RankNoveltyValidator
from watts.evolution.replacement_strategy import _release


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='foo', help='exp name')
parser.add_argument("--args_file", type=str, default=os.path.join('sample_args', 'walker_args.yaml'), help='path to args file')
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
    args.exp_name = 'poet'

    registry = Registrar(file_args=args)
    # game_schema = GameSchema(registry.gdy_file) # Used for GraphValidator
    wrappers = add_wrappers(args.wrappers)
    # gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    gym_factory = WalkerFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info)


    # generator = StaticGenerator(args.initial_level_string)
    # generator = EvolutionaryGenerator(args.initial_level_string,
    #                                   file_args=registry.get_generator_config)
    generator = WalkerConfigGenerator(**registry.get_generator_config)

    archive_dict = OrderedDict()

    s = SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                 trainer_config=registry.get_trainer_config,
                                 registered_gym_name=registry.env_name,
                                 network_factory=network_factory,
                                 gym_factory=gym_factory,
                                 log_id=f"{_args.exp_name}_{0}")

    if args.use_snapshot:
        manager = POETManagerSerializer.deserialize()
    else:
        manager = PoetManager(exp_name=_args.exp_name,
                              gym_factory=gym_factory,
                              network_factory=network_factory,
                              initial_pair=Pairing(solver=s,
                                                   generator=generator),
                              evolution_strategy=POETStrategy(level_validator=ParentCutoffValidator(env_config=registry.get_config_to_build_rllib_env,
                                                                                                    low_cutoff=15,
                                                                                                    high_cutoff=250),
                                                              replacement_strategy=ReplaceOldest(max_pairings=args.max_envs,
                                                                                                 archive=archive_dict),
                                                              selection_strategy=SelectRandomly(args.max_children),
                                                              transfer_strategy=GetBestZeroOrOneShotSolver(ZeroShotCartesian(config=registry.get_config_to_build_rllib_env),
                                                                                             default_trainer_config=registry.get_trainer_config),
                                                              # these are fed to the RankNoveltyValidator
                                                              # for now this validator is explicitly coded into
                                                              # the POETStrategy
                                                              env_config=registry.get_config_to_build_rllib_env,
                                                              agent_make_fn=network_factory.make(),
                                                              env_make_fn=gym_factory.make(),
                                                              historical_archive=archive_dict,
                                                              density_threshold=1.,
                                                              k=5,
                                                              low_cutoff=50,
                                                              high_cutoff=300.,
                                                              mutation_rate=args.mutation_rate),
                              transfer_strategy=GetBestZeroOrOneShotSolver(ZeroShotCartesian(config=registry.get_config_to_build_rllib_env),
                                                                                             default_trainer_config=registry.get_trainer_config),
                              registrar=registry)

    try:
        manager.run()
        print("finished algorithm")
    except KeyboardInterrupt as e:
        error = e
        print('_'*40)
        print(error)
        print('_'*40)
    finally:
        _release(archive_dict, manager.active_population)
        archive_dict['run_stats'] = manager.stats
        archive_dict['tournament_stats'] = manager._transfer_strategy.internal_transfer_strategy.tournaments
        save_obj(archive_dict,
                 os.path.join('..', 'enigma_logs', _args.exp_name),
                 'total_serialized_alg')
        
        elapsed = time.time() - start
        print(elapsed // 60, " minutes")
        # print(f"{len(manager.active_population)} PAIR objects: \n {manager.active_population}")
        ray.shutdown()
