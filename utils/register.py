import os

import gym
from gym.spaces import MultiDiscrete, Discrete


import griddly
from griddly import gd
from griddly.util.rllib.wrappers.core import RLlibEnv

from ray.rllib.agents import ppo, impala, es, maml, sac, ddpg, dqn


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

        if self.file_args.pictures:
            self.observer = gd.ObserverType.SPRITE_2D
        else:
            self.observer = gd.ObserverType.VECTOR

        self.gdy_file = os.path.join(self.file_args.lvl_dir, f'{self.file_args.game}.yaml')

        self.rllib_env_config = {'environment_name': self.name,
                                 'yaml_file': self.gdy_file,
                                 'level': self.file_args.init_lvl,
                                 'max_steps': self.file_args.game_len,
                                 'global_observer_type': self.observer,
                                 'player_observer_type': self.observer,
                                 'random_level_on_reset': False,
                                 'record_video_config': {
                                       'frequency': self.file_args.record_freq
                                    }
                                 }

        env = RLlibEnv(self.rllib_env_config)
        _ = env.reset()
        self.act_space = env.action_space
        self.obs_space = env.observation_space
        self.n_actions = 0

        # In Griddly, the zero-th action of each Discrete action in a no-op.
        # This means that there are two no-ops in Zelda. Therefore, we just manually count the
        # operations for each action
        if type(self.act_space) == MultiDiscrete:
            for k in self.act_space.nvec:
                self.n_actions += k
            # self.n_actions = prod(self.act_space.nvec)
        elif type(self.act_space) == Discrete:
            self.n_actions += self.act_space.n
        else:
            raise ValueError(f"Unsupported action type in game: {file_args.game}. "
                             f"Only Discrete and MultiDiscrete are supported")

        env.game.release()
        del env

        self.generator_config = {

        }

        self.nn_build_config = {
                'action_space': self.act_space,
                'obs_space': self.obs_space,
                'model_config': {},
                'num_outputs': self.n_actions,
                'name': self.file_args.network_name
        }

        # Trainer Config for selected algorithm
        self.trainer_config, self.trainer_constr = get_default_trainer_config_and_constructor(self.file_args.opt_algo)

        self.trainer_config['env_config'] = self.rllib_env_config
        self.trainer_config['env'] = self.name
        self.trainer_config["model"] = {
                'custom_model': self.file_args.network_name,
                'custom_model_config': {}
            }
        self.trainer_config["framework"] = self.file_args.framework
        self.trainer_config["num_workers"] = 2
        self.trainer_config["num_envs_per_worker"] = 2

    @property
    def get_nn_build_info(self):
        return self.nn_build_config

    @property
    def get_trainer_config(self):
        return self.trainer_config

    @property
    def get_config_to_build_rllib_env(self):
        return self.rllib_env_config

    @property
    def env_name(self):
        return self.name

    @property
    def network_name(self):
        return self.file_args.network_name

    @property
    def get_generator_config(self):
        return self.generator_config


if __name__ == "__main__":
    from utils.loader import load_from_yaml
    import os
    os.chdir('..')
    arg_path = os.path.join('args.yaml')
    file_args = load_from_yaml(arg_path)

    Registry = Registrar(file_args)
    print(Registry.env_name)
    print(Registry.get_nn_build_info)
    print(Registry.get_config_to_build_rllib_env)

    env = RLlibEnv(Registry.get_config_to_build_rllib_env)
    state = env.reset()
    ns, r, d, info = env.step(env.action_space.sample())
    print(info)
