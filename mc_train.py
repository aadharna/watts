import argparse
import os
import ray
import sys
import time

from watts.game.GameSchema import GameSchema
from watts.generators.AIIDE_generator import EvolutionaryGenerator
from watts.gym_factory import GridGameFactory
from watts.network_factory import NetworkFactory
from watts.managers.MCManager import MCManager
from watts.solvers.SingleAgentSolver import SingleAgentSolver
from watts.utils.gym_wrappers import add_wrappers
from watts.utils.register import Registrar
from watts.utils.loader import load_from_yaml, save_obj
from watts.validators.level_validator import AlwaysValidator, RandomVariableValidator
from watts.validators.graph_validator import GraphValidator
from watts.validators.agent_validator import RandomAgentValidator, ParentCutoffValidator, PositiveGAEValidator
from watts.validators.Deepmind_validator import DeepMindFullValidator, DeepMindAppendixValidator
from watts.validators.complex_coevo_validator import Foo
from watts.validators.PINSKY_validator import PINSKYValidator


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='foo.dma', help='exp name')
parser.add_argument("--args_file", type=str, default=os.path.join('sample_args', 'args_mc_exp.yaml'),
                    help='path to args file')
_args = parser.parse_args()


if __name__ == "__main__":
    """
    This file implements the algorithm explored in the appendix of 
    Open-Ended Learning Leads to Generally Capable Agents
    See pages 40-41 the World-agent co-evolution section
    
    This file should be provided a name that is used to track this experiment 
      as well as an arguments file.
    """

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1,
             ignore_reinit_error=True,)
             # log_to_driver=False,
             # local_mode=True)

    args = load_from_yaml(fpath=_args.args_file)
    args.exp_name = f'dma.{_args.exp_name}' if 'dma' not in _args.exp_name else _args.exp_name

    start = time.time()

    registry = Registrar(file_args=args)
    game_schema = GameSchema(registry.gdy_file)  # Used for GraphValidator

    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info)
    level_string = '''w w w w w w w w w w w w w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . t w\nw . . . . . w . . . . . w\nw . x . . . . . . . . . w\nw w w w w w w w w w w w w\n'''

    generator = EvolutionaryGenerator(level_string=level_string,
                                      file_args=registry.get_generator_config)

    agent = SingleAgentSolver.remote(registry.trainer_constr,
                                     registry.get_trainer_config,
                                     registered_gym_name=registry.name,
                                     network_factory=network_factory,
                                     gym_factory=gym_factory,
                                     weights={},
                                     log_id=f"{_args.exp_name}_0")

    if args.valType == 'always':
        val = AlwaysValidator()
    elif args.valType == 'PCV':
        val = ParentCutoffValidator(registry.get_config_to_build_rllib_env,
                                    low_cutoff=args.val_config['low'],
                                    high_cutoff=args.val_config['high'],
                                    n_repeats=args.val_config['n_repeats'])
    elif args.valType == 'RA':
        val = RandomAgentValidator(network_factory_monad=network_factory.make(),
                                   env_config=registry.get_config_to_build_rllib_env,
                                   low_cutoff=args.val_config['low'],
                                   high_cutoff=args.val_config['high'],
                                   n_repeats=args.val_config['n_repeats'])
    elif args.valType == 'RV':
        val = RandomVariableValidator()
    elif args.valType == 'Graph':
        val = GraphValidator(game_schema=game_schema)
    elif args.valType == 'Pinsky':
        val = PINSKYValidator(network_factory_monad=network_factory.make(),
                              env_config=registry.get_config_to_build_rllib_env,
                              low_cutoff=args.val_config['low'],
                              high_cutoff=args.val_config['high'],
                              n_repeats=args.val_config['n_repeats'],
                              game_schema=game_schema)
    elif args.valType == 'DeepmindFull':
        val = DeepMindFullValidator(network_factory_monad=network_factory.make(),
                                    env_config=registry.get_config_to_build_rllib_env,
                                    low_cutoff=args.val_config['low'],
                                    high_cutoff=args.val_config['high'],
                                    n_tasks_parent_greater_than_high=args.val_config['n_tasks_parent_greater_than_high'],
                                    n_tasks_difference_greater_than_margin=args.val_config['n_tasks_difference_greater_than_margin'],
                                    margin=args.val_config['margin'],
                                    n_repeats=args.val_config['n_repeats'])
    elif args.valType == 'DeepmindAppendix':
        val = DeepMindAppendixValidator(env_config=registry.get_config_to_build_rllib_env,
                                        low_cutoff=args.val_config['low'],
                                        n_repeats=args.val_config['n_repeats'])
    elif args.valType == 'Foo':
        val = Foo(env_config=registry.get_config_to_build_rllib_env,
                  low_cutoff=args.val_config['low'],
                  high_cutoff=args.val_config['high'],
                  n_repeats=args.val_config['n_repeats'],
                  game_schema=game_schema,
                  network_factory_monad=network_factory.make())
    elif args.valType == "gae":
        val = PositiveGAEValidator(env_config=registry.get_config_to_build_rllib_env,
                                   n_repeats=args.val_config['n_repeats'])
    else:
        raise ValueError('pick new validator')

    manager = MCManager(exp_name=_args.exp_name,
                        reproduction_limit=args.reproduction_limit,
                        mutation_timer=args.evolution_timer,
                        n_children=args.n_children,
                        snapshot_timer=args.snapshot_timer,
                        mutation_rate=args.mutation_rate,
                        agent=agent,
                        generator=generator,
                        validator=val,
                        gym_factory=gym_factory,
                        network_factory=network_factory,
                        registrar=registry)

    try:
        manager.run()
        print("finished algorithm")
    except (Exception, KeyboardInterrupt) as e:
        error = e
        print('_' * 40)
        print(error)
        print('_' * 40)
    finally:
        foo = {
            'run_stats': manager.stats,
        }
        manager.agent.release.remote()
        save_obj(foo,
                 os.path.join('..', 'enigma_logs', _args.exp_name),
                 'MC_total_serialized_alg')

        elapsed = time.time() - start
        print(elapsed // 60, " minutes")
        ray.shutdown()
