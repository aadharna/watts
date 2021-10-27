import argparse
import copy
import griddly
from griddly import gd
from griddly.util.rllib.environment.core import RLlibEnv, RLlibMultiAgentWrapper
import gym
from gym.spaces import MultiDiscrete, Discrete
import os
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.agents import ppo, impala, es, maml, sac, ddpg, dqn
from ray.rllib.utils import add_mixins
from utils.trainer_reset import ResetConfigOverride
from pprint import pprint


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

def policy_mapping_function(policies):
    def _map(agent_id):
        for policy in policies.keys():
            if agent_id.start_with(policy):
                return policy

    return _map

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
                                 'global_observer_type': self.observer,
                                 'player_observer_type': self.observer,
                                 'random_level_on_reset': False,
                                 'record_video_config': {
                                       'frequency': self.file_args.record_freq,
                                       'directory': os.path.join('.', 'videos')
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
        
        self.n_actions = 0

        # In Griddly, the zero-th action of each Discrete action in a no-op.
        # This means that there are two no-ops in Zelda. Therefore, we just manually count the
        # operations for each action
        if type(self.act_space) == MultiDiscrete:
            self.n_actions = sum(self.act_space.nvec)
        elif type(self.act_space) == Discrete:
            self.n_actions = self.act_space.n
        else:
            raise ValueError(f"Unsupported action type in game: {file_args.game}. "
                             f"Only Discrete and MultiDiscrete are supported")

        env.game.release()
        del env

        # todo ??
        # We might want to make this into something better.
        #  Also, this should probably have it's own special prepare
        #  function since different generators might require different initialization arguments
        #   Also, this is being put into an argparse.Namespace because that makes the
        #    code in the generator readable, but that should be switched to a dict for consistency.
        self.generator_config = argparse.Namespace(**{
            'mechanics': self.file_args.mechanics,
            'singletons': self.file_args.singletons,
            'at_least_one': self.file_args.at_least_one,
            'immortal': self.file_args.immortal,
            'floor': self.file_args.floor,
            'probs': self.file_args.probs,
        })

        if 'MultiAgent' not in self.file_args.wrappers:
            self.nn_build_config = {
                    'action_space': self.act_space,
                    'obs_space': self.obs_space,
                    'model_config': {},
                    'num_outputs': self.n_actions,
                    'agent': 'default',
                    'network_name': self.file_args.network_name
            }
        else:
            self.nn_build_config = []
            for policy in self.file_args.policies:
                self.nn_build_config.append({
                    'action_space': self.act_space,
                    'obs_space': self.obs_space,
                    'model_config': {},
                    'num_outputs': self.n_actions,
                    'agent': policy['agent'],
                    'network_name': policy['network_name']
                    })


        # Trainer Config for selected algorithm
        self.trainer_config, self.trainer_constr = get_default_trainer_config_and_constructor(self.file_args.opt_algo)
        if self.file_args.custom_trainer_config_override:
            self.trainer_constr = add_mixins(self.trainer_constr, [ResetConfigOverride])

        self.trainer_config['env_config'] = self.rllib_env_config
        self.trainer_config['env'] = self.name
        self.trainer_config["framework"] = self.file_args.framework
        self.trainer_config["num_workers"] = 1
        self.trainer_config["num_envs_per_worker"] = 2
        self.trainer_config['simple_optimizer'] = True
        if 'MultiAgent' in self.file_args.wrappers:
            self.trainer_config['multiagent'] = {}
            self.trainer_config['multiagent']['policies'] = {}
            policies = {}
            for policy in self.file_args.policies:
                model_config = {
                        'model': {
                            'custom_model': policy['network_name'],
                            'custom_model_config': {}
                            }
                        }
                policy_config = (
                            None,
                            self.obs_space,
                            self.act_space,
                            model_config
                            # How can we support different obs/act spaces for different agents?
                        )
                policies[policy['agent']] =  policy_config
            self.trainer_config['multiagent'] = {
                    'policies': {**policies},
                    'policy_mapping_fn': policy_mapping_function(self.file_args.policies)
                    
                    }
        else:
            self.trainer_config["model"] = {
                    'custom_model': self.file_args.network_name,
                    'custom_model_config': {}
                }
        # self.trainer_config['log_level'] = 'INFO'
        # self.trainer_config['num_gpus'] = 0.1

    def __build_nn_config(self):
        if 'policies' not in self.file_args:
            return {
                    'action_space': self.act_space,
                    'obs_space': self.obs_space,
                    'model_config': {},
                    'num_outputs': self.n_actions,
                    'name': self.file_args.network_name
            }

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
