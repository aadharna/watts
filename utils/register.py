import os

import gym
from gym.spaces import MultiDiscrete, Discrete

from numpy import prod

import griddly
from griddly import gd
from griddly.util.rllib.wrappers.core import RLlibEnv

from ray.tune.registry import register_env

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
        register_env(self.name, RLlibEnv)

        if self.file_args.pictures:
            self.observer = gd.ObserverType.SPRITE_2D
        else:
            self.observer = gd.ObserverType.VECTOR

        self.gdy_file = os.path.join(file_args.lvl_dir, f'{self.file_args.game}.yaml')

        self.trainer_config = {'environment_name': self.name,
                               'yaml_file': self.gdy_file,
                               'level': self.file_args.init_lvl,
                               'max_steps': self.file_args.game_len,
                               'global_observer_type': self.observer,
                               'player_observer_type': self.observer,
                               'record_video_config': {
                                       'frequency': self.file_args.record_freq
                                   }
                               }

        env = RLlibEnv(self.trainer_config)
        _ = env.reset()
        self.act_space = env.action_space
        self.obs_space = env.observation_space
        if type(self.act_space) == MultiDiscrete:
            self.n_actions = prod(self.act_space.nvec)
        elif type(self.act_space) == Discrete:
            self.n_actions = self.act_space.n
        else:
            raise ValueError(f"Unsupported action type in game: {file_args.game}. "
                             f"Only Discrete and MultiDiscrete are supported")

        env.game.release()
        del env

        self.information_dict = {
            'env_name': self.name,
            'trainer_config': self.trainer_config,
            'nn_build_config': {
                'action_space': self.act_space,
                'obs_space': self.obs_space,
                'model_config': {},
                'num_outputs': self.n_actions,
                'name': self.file_args.network_name
            }
        }


    @property
    def get_nn_build_info(self):
        return self.information_dict['nn_build_config']

    @property
    def get_rllib_config(self):
        return self.information_dict['trainer_config']

    @property
    def env_name(self):
        return self.information_dict['env_name']

    @property
    def network_name(self):
        return self.information_dict['nn_build_config']['name']


if __name__ == "__main__":
    from utils.loader import load_from_yaml
    import os
    os.chdir('..')
    arg_path = os.path.join('args.yaml')
    file_args = load_from_yaml(arg_path)

    Registry = Registrar(file_args)
    print(Registry.env_name)
    print(Registry.get_nn_build_info)
    print(Registry.get_rllib_config)

    env = RLlibEnv(Registry.get_rllib_config)
    state = env.reset()
    ns, r, d, info = env.step(env.action_space.sample())
    print(info)
