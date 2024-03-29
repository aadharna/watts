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
parser.add_argument("--exp_name", type=str, default='foo.poet', help='exp name')
parser.add_argument("--args_file", type=str, default=os.path.join('sample_args', 'args.yaml'), help='path to args file')
_args = parser.parse_args()


if __name__ == "__main__":
    """
    This file launches the POET (and therefore also PINSKY) algorithms. 
    This file should be provided a name that is used to track this experiment 
      as well as an arguments file.
    """

    # make the python that launched this script the version of python
    # that all ray-processes will also use.
    # this is very important
    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    # initialize ray
    ray.init(num_gpus=1,
             ignore_reinit_error=True,)
             # log_to_driver=False,
             # local_mode=True)

    start = time.time()

    # load arguments from file
    args = load_from_yaml(fpath=_args.args_file)
    # save the name into the loaded-file-arguments and tag it as poet
    args.exp_name = f'poet.{_args.exp_name}' if 'poet' not in _args.exp_name else _args.exp_name

    # extract and setup experiment info from the arguments
    registry = Registrar(file_args=args)

    # game_schema = GameSchema(registry.gdy_file) # Used for GraphValidator
    # get any classes used the wrap the Gym/RLliv env
    wrappers = add_wrappers(args.wrappers)

    # register the gym with rllib and create the factory that will build new envs on demand
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    # gym_factory = WalkerFactory(registry.env_name, env_wrappers=wrappers)

    # register the neural network with rllib and create the factory that will build
    #  new NNs on demand
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info,
                                     policy_class=registry.policy_class)

    # Define a Generator class that will build/evolve/train/sample/generate new learning environemnts
    generator = EvolutionaryGenerator(args.initial_level_string,
                                      file_args=registry.get_generator_config)
    # generator = WalkerConfigGenerator(**registry.get_generator_config)
    # generator = StaticGenerator(args.initial_level_string)

    # dict to save agent/environment information
    archive_dict = OrderedDict()

    # This solver will define optimization for the agent's NN.
    # this specific solver, the Single-Agent-Solver is a general Solver class
    # that can call into various RLlib optimization algorithms.
    s = SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                 trainer_config=registry.get_trainer_config,
                                 registered_gym_name=registry.env_name,
                                 network_factory=network_factory,
                                 gym_factory=gym_factory,
                                 log_id=f"{args.exp_name}_{0}")

    if args.use_snapshot:
        manager = POETManagerSerializer.deserialize()
    else:
        # Define the POETManager!
        manager = PoetManager(exp_name=args.exp_name,
                              gym_factory=gym_factory,
                              network_factory=network_factory,
                              # combine the agent and generator into a Pair!
                              initial_pair=Pairing(solver=s,
                                                   generator=generator),
                              # define how the learning environemnts evolve over time
                              evolution_strategy=POETStrategy(
                                  # Define how new learning environments get validated as
                                  #  being ``good'' to train on
                                  level_validator=ParentCutoffValidator(env_config=registry.get_config_to_build_rllib_env,
                                                                        low_cutoff=1,
                                                                        high_cutoff=450,
                                                                        n_repeats=1),
                                  # Define how to remove ``old'' pairs
                                  replacement_strategy=ReplaceOldest(max_pairings=args.max_envs,
                                                                     archive=archive_dict),
                                  # Define how to pick which pairs will have children
                                  selection_strategy=SelectRandomly(args.max_children),
                                  # Define how the newly created agent
                                  #  gets its weights
                                  #   this transfer step to find good weights is specific to the POETManager
                                  transfer_strategy=GetBestZeroOrOneShotSolver(ZeroShotCartesian(config=registry.get_config_to_build_rllib_env),
                                                                 default_trainer_config=registry.get_trainer_config),
                                  # these are fed to the RankNoveltyValidator
                                  # for now this validator is explicitly coded into
                                  # the POETStrategy
                                  env_config=registry.get_config_to_build_rllib_env,
                                  network_factory=network_factory,
                                  env_factory=gym_factory,
                                  historical_archive=archive_dict,
                                  density_threshold=1.,
                                  k=5,
                                  low_cutoff=1,
                                  high_cutoff=250,
                                  mutation_rate=args.mutation_rate),
                              # Define how to goal-switch agents between different learning environments
                              transfer_strategy=GetBestZeroOrOneShotSolver(ZeroShotCartesian(config=registry.get_config_to_build_rllib_env),
                                                                                             default_trainer_config=registry.get_trainer_config),
                              registrar=registry,
                              archive_dict=archive_dict)

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
        # this is because the GetBestZeroOrOneShotSolver strategy wraps a GetBestSolver strategy
        archive_dict['tournament_stats'] = manager._transfer_strategy.internal_transfer_strategy.tournaments
        save_obj(archive_dict,
                 os.path.join('.', 'watts_logs', args.exp_name),
                 'total_serialized_alg')
        
        elapsed = time.time() - start
        print(elapsed // 60, " minutes")
        # print(f"{len(manager.active_population)} PAIR objects: \n {manager.active_population}")
        ray.shutdown()
