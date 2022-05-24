import gym
import os
import ray
import sys
from typing import Optional, Dict
import numpy as np

from griddly.util.rllib.environment.core import RLlibEnv
from ray.tune import tune, logger
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch


from watts.models.AIIDE_network import AIIDEActor
from watts.models.PCGRL_network import PCGRLAdversarial
from watts.utils.register import Registrar
from watts.utils.loader import load_from_yaml
from watts.utils.gym_wrappers import AlignedReward, Regret, PlacePredefinedSequence


if __name__ == "__main__":

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1, ignore_reinit_error=True)  # , log_to_driver=False, local_mode=True)

    args = load_from_yaml(os.path.join('sample_args', 'args.yaml'))
    args.exp_name = 'paired'
    args.network_name = 'Adversarial_PCGRL'

    registry = Registrar(file_args=args)

    config = registry.get_config_to_build_rllib_env
    config['board_shape'] = (15, 15)
    config['builder_max_steps'] = 50
    config['player_max_steps'] = 250


    def make_env(config):
        env = RLlibEnv(config)
        h_env = Regret(env, config)
        h_env = PlacePredefinedSequence(h_env, config)
        return h_env

    def make_unconstrained_paired(config):
        env = RLlibEnv(config)
        h_env = Regret(env, config)
        return h_env

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith('antagonist'):
            return 'antagonist'
        elif agent_id.startswith('protagonist'):
            return 'protagonist'
        else:
            return 'builder'


    ModelCatalog.register_custom_model('AIIDE', AIIDEActor)
    ModelCatalog.register_custom_model('PCGRL', PCGRLAdversarial)
    register_env('h_maze', make_env)

    h_env = make_env(config)
    _ = h_env.reset()
    config2 = {
        'env': 'h_maze',
        'num_workers': 4,
        "num_envs_per_worker": 4,
        "train_batch_size": 8192,
        'sgd_minibatch_size': 512,
        'env_config': config,
        # 'callbacks': PairedLevelExtractorCallback,
        'multiagent': {
            'policies': {
                'builder': (None,
                            h_env.observation_space['builder'],
                            h_env.action_space['builder'],
                            {'model': {'custom_model': 'PCGRL',
                                       'custom_model_config': {'cell_size': 2704}}}),
                'antagonist': (None,
                               h_env.observation_space['antagonist'],
                               h_env.action_space['antagonist'],
                               {'model': {'custom_model': 'PCGRL',
                                          'custom_model_config': {'cell_size': 144}}}),
                'protagonist': (None,
                                h_env.observation_space['protagonist'],
                                h_env.action_space['protagonist'],
                                {'model': {'custom_model': 'PCGRL',
                                           'custom_model_config': {'cell_size': 144}}})
            },
            'policy_mapping_fn': policy_mapping_fn
        },
        "framework": 'torch',
        "num_gpus": 1
    }

    stop = {"timesteps_total": 25000000}

    try:
        results = tune.run(PPOTrainer, config=config2, stop=stop,
                           local_dir=os.path.join('.', 'watts_logs'), checkpoint_at_end=True,
                           checkpoint_freq=200,
                           # restore='/home/aaron/Documents/watts/watts_logs/PPO_2022-02-22_10-00-42/PPO_h_maze_34657_00000_0_2022-02-22_10-00-42/checkpoint_003800/checkpoint-3800'
                           # callbacks=[SnapshotLoggerCallback()]
                           )
        # print(results.get_best_logdir(metric='episode_reward_mean', mode="max"))
    except (KeyboardInterrupt, Exception) as e:
        print(e)
    finally:
        print(results.get_best_checkpoint(trial=results.get_best_trial(metric='episode_reward_mean', mode='max'),
                                    metric='episode_reward_mean', mode='max'))
        print(results.get_last_checkpoint())
        ray.shutdown()


    # 25M step train result:
    # D:\PycharmProjects\thesis\enigma\watts_logs\PPO_2022-02-11_14-06-09\PPO_h_maze_abd19_00000_0_2022-02-11_14-06-09
