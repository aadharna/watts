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

    # class SnapshotLoggerCallback(TBXLoggerCallback):
    #     def __init__(self):
    #         super(SnapshotLoggerCallback, self).__init__()
    #         self.first_batch = True
    #
    #     def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
    #         super().log_trial_result(iteration, trial, result)
    #         picture_folder = os.path.join(trial.logdir, "images")
    #         if not os.path.exists(picture_folder):
    #             os.mkdir(picture_folder)
    #         try:
    #             image_list = result['episode_media']['level']
    #             image_list = np.array(image_list).squeeze()
    #             # print(f"{image_list.shape} items with {image_list.size} size")
    #             images = [image for image in image_list if (len(image) != 0 and image.size != 0)]
    #         except (KeyError, TypeError, IndexError) as e:
    #             return
    #         # save the first rollout
    #         if self.first_batch and len(images) != 0:
    #             self.first_batch = False
    #             np.save(os.path.join(picture_folder, str(iteration)) + '.npy', images)
    #         # do not save if iteration is not mod 10
    #         # do not save if there is nothing to save
    #         if not iteration % 10 == 0 or len(images) == 0:
    #             return
    #         else:
    #             np.save(os.path.join(picture_folder, str(iteration)) + '.npy', images)
    #
    #
    # class PairedRewardRewriter(DefaultCallbacks):
    #     def __init__(self):
    #         super().__init__()
    #
    #     def on_postprocess_trajectory(
    #         self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
    #         agent_id: AgentID, policy_id: PolicyID,
    #         policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
    #         original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
    #         """Called immediately after a policy's postprocess_fn is called.
    #
    #         You can use this callback to do additional postprocessing for a policy,
    #         including looking at the trajectory data of other agents in multi-agent
    #         settings.
    #
    #         Args:
    #             worker (RolloutWorker): Reference to the current rollout worker.
    #             episode (MultiAgentEpisode): Episode object.
    #             agent_id (str): Id of the current agent.
    #             policy_id (str): Id of the current policy for the agent.
    #             policies (dict): Mapping of policy id to policy objects. In single
    #                 agent mode there will only be a single "default" policy.
    #             postprocessed_batch (SampleBatch): The postprocessed sample batch
    #                 for this agent. You can mutate this object to apply your own
    #                 trajectory postprocessing.
    #             original_batches (dict): Mapping of agents to their unpostprocessed
    #                 trajectory data. You should not mutate this object.
    #             kwargs: Forward compatibility placeholder.
    #         """
    #         pass
    #
    # class PairedLevelExtractorCallback(DefaultCallbacks):
    #     def __init__(self):
    #         super().__init__()
    #
    #     def on_episode_end(self,
    #                    *,
    #                    worker: "RolloutWorker",
    #                    base_env: BaseEnv,
    #                    policies: Dict[PolicyID, Policy],
    #                    episode: MultiAgentEpisode,
    #                    env_index: Optional[int] = None,
    #                    **kwargs) -> None:
    #         """Runs when an episode is done.
    #
    #         Args:
    #             worker (RolloutWorker): Reference to the current rollout worker.
    #             base_env (BaseEnv): BaseEnv running the episode. The underlying
    #                 env object can be gotten by calling base_env.get_unwrapped().
    #             policies (dict): Mapping of policy id to policy objects. In single
    #                 agent mode there will only be a single "default" policy.
    #             episode (MultiAgentEpisode): Episode object which contains episode
    #                 state. You can use the `episode.user_data` dict to store
    #                 temporary data, and `episode.custom_metrics` to store custom
    #                 metrics for the episode.
    #             env_index (EnvID): Obsoleted: The ID of the environment, which the
    #                 episode belongs to.
    #             kwargs: Forward compatibility placeholder.
    #         """
    #         envs = base_env.get_unwrapped()
    #         gifs = []
    #         for e in envs:
    #             gv = e.global_view
    #             if gv is not None:
    #                 gifs.append(gv)
    #         if len(gifs) == 0:
    #             return
    #         else:
    #             episode.media['level'] = gifs

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
