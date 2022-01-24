import copy
import argparse

import gym.spaces
from griddly import gd
from griddly.util.rllib.environment.core import RLlibEnv, RLlibMultiAgentWrapper

from gym.spaces import MultiDiscrete, Discrete, Box
import numpy as np
import os
from ray.rllib.agents import ppo, impala, maml, sac, ddpg, dqn
from ..agents import es
from ray.rllib.utils import add_mixins

from ..utils.trainer_reset import ResetConfigOverride
from ..utils.gym_wrappers import HierarchicalBuilderEnv


def get_default_trainer_config_and_constructor(opt_algo):
    if opt_algo == "OpenAIES":
        return es.DEFAULT_CONFIG.copy(), es.ESTrainer
    elif opt_algo == "PPO":
        return ppo.DEFAULT_CONFIG.copy(), ppo.PPOTrainer
    elif opt_algo == 'MAML':
        return maml.DEFAULT_CONFIG.copy(), maml.MAMLTrainer
    elif opt_algo == 'DDPG':
        return ddpg.DEFAULT_CONFIG.copy(), ddpg.DDPGTrainer
    elif opt_algo == 'DQN':
        return dqn.DEFAULT_CONFIG.copy(), dqn.DQNTrainer
    elif opt_algo == 'SAC':
        return sac.DEFAULT_CONFIG.copy(), sac.SACTrainer
    elif opt_algo == 'IMPALA':
        return impala.DEFAULT_CONFIG.copy(), impala.ImpalaTrainer
    else:
        raise ValueError('Pick another opt_algo')


class Registrar:
    id = 0

    def __init__(self, file_args):
        """The Registrar handles storing and dispensing information needed in factory make functions.
            e.g. action_space info, observer info, env name, etc.

        :param file_args:
        """
        if not Registrar.id == 0:
            raise ValueError("class Registrar is a singleton")

        # increment counter to stop this class from being built again
        Registrar.id += 1

        self.file_args = file_args
        self.name = f'{self.file_args.game}_custom'
        genType = self.file_args.generatorType

        if self.file_args.engine == 'GDY':
            if self.file_args.pictures:
                self.observer = gd.ObserverType.SPRITE_2D
            else:
                self.observer = gd.ObserverType.VECTOR

            self.gdy_file = os.path.join(self.file_args.lvl_dir, f'{self.file_args.game}.yaml')
            self.base_path = os.getcwd()

            self.rllib_env_config = {'environment_name': self.name,
                                     'yaml_file': os.path.join(self.base_path, self.gdy_file),
                                     'level': self.file_args.init_lvl,
                                     'max_steps': self.file_args.game_len,
                                     'global_observer_type': gd.ObserverType.BLOCK_2D,
                                     'player_observer_type': self.observer,
                                     'random_level_on_reset': False,
                                     'record_video_config': {
                                           'frequency': self.file_args.record_freq,
                                           'directory': os.path.join('videos',
                                                                     f'{self.file_args.exp_name}_{self.name}_{self.file_args.generatorType}_{self.file_args.network_name}'),
                                           'include_global': True,
                                        }
                                     }

            env = RLlibEnv(self.rllib_env_config)
            if 'MultiAgent' in self.file_args.wrappers:
                env = RLlibMultiAgentWrapper(env, self.rllib_env_config)
            state = env.reset()
            self.act_space = env.action_space
            self.obs_space = env.observation_space

            if isinstance(state, dict):
                agent_keys = list(state.keys())

            # todo ??
            # We might want to make this into something better.
            #  Also, this should probably have it's own special prepare
            #  function since different generators might require different initialization arguments
            #   Also, this is being put into an argparse.Namespace because that makes the
            #    code in the generator readable, but that should be switched to a dict for consistency.
            if genType == 'evolutionary':
                self.generator_config = argparse.Namespace(**{
                    'mechanics': self.file_args.mechanics,
                    'singletons': self.file_args.singletons,
                    'at_least_one': self.file_args.at_least_one,
                    'immortal': self.file_args.immortal,
                    'floor': self.file_args.floor,
                    'probs': self.file_args.probs,
                })
            elif genType == 'pcgrl':

                e2 = HierarchicalBuilderEnv(env, self.rllib_env_config)
                self.generator_config = {
                    'action_space': e2.builder_env.action_space,
                    'obs_space': Box(0.0, 255.0, e2.builder_env.action_space.nvec[:3], np.float64),
                    'model_config': self.file_args.model_config,
                    'num_outputs': sum(e2.builder_env.action_space.nvec),
                    'name': 'builder'
                }
                e2.builder_env.game.release()
                del e2

            env.game.release()
            del env

        elif self.file_args.engine == 'box2d':
            from watts.utils.box2d.biped_walker_custom import BipedalWalkerCustom, DEFAULT_ENV
            from watts.utils.box2d.walker_wrapper import OverrideWalker
            self.rllib_env_config = {
                'config_tuple': DEFAULT_ENV,
                'level_string': str(DEFAULT_ENV)
            }
            env = BipedalWalkerCustom(DEFAULT_ENV)
            env = OverrideWalker(env)
            self.act_space = env.action_space  # Box space with shape 4
            self.obs_space = env.observation_space  # Box space with shape 24
            del env

            if genType == 'walker':
                self.generator_config = {'parent_env_config': DEFAULT_ENV,
                                         'categories': ('stump', 'pit', 'roughness', 'stair')}

        self.n_actions = 0
        # In Griddly, the zero-th action of each Discrete action in a no-op.
        # This means that there are two no-ops in Zelda. Therefore, we just manually count the
        # operations for each action
        if type(self.act_space) == MultiDiscrete:
            self.n_actions = sum(self.act_space.nvec)
        elif type(self.act_space) == Discrete:
            self.n_actions = self.act_space.n
        elif isinstance(self.act_space, gym.spaces.Box):
            if self.file_args.opt_algo == 'OpenAIES':
                # for some reason the OpenAIES networks don't
                #  create mu, sigma vectors and only do mu. Therefore, we don't need 2 * action_space.shape[0]
                self.n_actions = self.act_space.shape[0]
            else:
                # we need logits for mu and sigma vectors.
                self.n_actions = 2 * self.act_space.shape[0]
            if self.file_args.engine == 'GDY':
                raise ValueError(f"Unsupported action type in game: {file_args.game}. "
                                 f"Only Discrete and MultiDiscrete are supported with {self.file_args.engine}")

        self.nn_build_config = {
            'action_space': self.act_space,
            'obs_space': self.obs_space,
            'model_config': self.file_args.model_config,
            'num_outputs': self.n_actions,
            'name': self.file_args.network_name
        }


        # Trainer Config for selected algorithm
        self.trainer_config, self.trainer_constr = get_default_trainer_config_and_constructor(self.file_args.opt_algo)
        if self.file_args.custom_trainer_config_override:
            self.trainer_constr = add_mixins(self.trainer_constr, [ResetConfigOverride])

        self.trainer_config['env_config'] = self.rllib_env_config
        self.trainer_config['env'] = self.name
        self.trainer_config["model"] = {
            'custom_model': self.file_args.network_name,
            'custom_model_config': {}
        }
        self.trainer_config["framework"] = self.file_args.framework
        self.trainer_config["num_workers"] = 1
        self.trainer_config["num_envs_per_worker"] = 2
        # self.trainer_config['simple_optimizer'] = True
        # self.trainer_config['log_level'] = 'INFO'
        # self.trainer_config['num_gpus'] = 0.03

    @property
    def get_nn_build_info(self):
        return copy.deepcopy(self.nn_build_config)

    @property
    def get_trainer_config(self):
        return copy.deepcopy(self.trainer_config)

    @property
    def get_config_to_build_rllib_env(self):
        return copy.deepcopy(self.rllib_env_config)

    @property
    def env_name(self):
        return self.name

    @property
    def network_name(self):
        return self.file_args.network_name

    @property
    def get_generator_config(self):
        return copy.deepcopy(self.generator_config)
