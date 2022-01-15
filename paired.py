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
from watts.utils.gym_wrappers import AlignedReward, Regret


if __name__ == "__main__":

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1, ignore_reinit_error=True)  # , log_to_driver=False, local_mode=True)

    args = load_from_yaml(os.path.join('sample_args', 'args.yaml'))
    args.exp_name = 'paired'

    registry = Registrar(file_args=args)

    config = registry.get_config_to_build_rllib_env
    config['board_shape'] = (15, 15)
    config['builder_max_steps'] = 50
    config['max_steps'] = 150


    def make_env(config):
        env = RLlibEnv(config)
        h_env = Regret(env, config)
        return h_env


    def policy_mapping_fn(agent_id):
        if agent_id.startswith('antagonist'):
            return 'antagonist'
        elif agent_id.startswith('protagonist'):
            return 'protagonist'
        else:
            return 'builder'


    ModelCatalog.register_custom_model('AIIDE', AIIDEActor)
    ModelCatalog.register_custom_model('PCGRL', PCGRLAdversarial)
    register_env('h_maze', make_env)

    class SnapshotLoggerCallback(TBXLoggerCallback):
        def __init__(self):
            super(SnapshotLoggerCallback, self).__init__()
            self.i = 0

        def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
            super().log_trial_result(iteration, trial, result)
            try:
                image = result['episode_media']['level'][0]
            except KeyError:
                return
            picture_folder = os.path.join(trial.logdir, "images")
            if not os.path.exists(picture_folder):
                os.mkdir(picture_folder)
            np.save(os.path.join(picture_folder, str(self.i)) + '.npy', image)
            self.i += 1


    class PairedLevelExtractorCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()

        def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
            """Runs when an episode is done.

            Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                    env object can be gotten by calling base_env.get_unwrapped().
                policies (dict): Mapping of policy id to policy objects. In single
                    agent mode there will only be a single "default" policy.
                episode (MultiAgentEpisode): Episode object which contains episode
                    state. You can use the `episode.user_data` dict to store
                    temporary data, and `episode.custom_metrics` to store custom
                    metrics for the episode.
                env_index (EnvID): Obsoleted: The ID of the environment, which the
                    episode belongs to.
                kwargs: Forward compatibility placeholder.
            """
            envs = base_env.get_unwrapped()
            episode.media['level'] = envs[0].global_view.swapaxes(0, 2).astype(float)

    h_env = make_env(config)
    _ = h_env.reset()
    config2 = {
        'env': 'h_maze',
        'num_workers': 1,
        "num_envs_per_worker": 1,
        'env_config': config,
        'callbacks': PairedLevelExtractorCallback,
        'multiagent': {
            'policies': {
                'builder': (None, h_env.builder_env.observation_space,
                            h_env.builder_env.action_space, {'model': {'custom_model': 'PCGRL',
                                                                       'custom_model_config': {'cell_size': 2704}}}),
                'antagonist': (None, h_env.env.observation_space,
                               h_env.env.action_space, {'model': {'custom_model': 'PCGRL',
                                                                  'custom_model_config': {'cell_size': 144}}}),
                'protagonist': (None, h_env.env.observation_space,
                                h_env.env.action_space, {'model': {'custom_model': 'PCGRL',
                                                                   'custom_model_config': {'cell_size': 144}}})
            },
            'policy_mapping_fn': policy_mapping_fn
        },
        "framework": 'torch',
        "num_gpus": 1
    }

    stop = {"timesteps_total": 2000000}

    try:
        results = tune.run(PPOTrainer, config=config2, stop=stop,
                           local_dir=os.path.join('.', 'watts_logs'), checkpoint_at_end=True,
                           callbacks=[SnapshotLoggerCallback()])
        print(results.best_logdir)
    except (KeyboardInterrupt, Exception) as e:
        print(e)
    finally:
        ray.shutdown()


