from typing import Dict, Optional

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch

from watts.solvers.base import BaseSolver
from watts.evaluators.rollout import rollout



class PAIREDAgentSolver(BaseSolver):
    def __init__(self, solvers):
        BaseSolver.__init__(self)

        # todo We might want to name these agents to access via keys
        #  rather than a convention of [protagonist, antagonist].

        # todo switch to having factories in here?
        self.solvers = solvers

    @staticmethod
    def evaluate(actors: list, env) -> dict:
        """We evaluate two agents and take the difference between their scores
            for the level-generator agent to maximize.

        :param actors: list of NNs to evaluate
        :param env: RLlibEnv environment to run the simulation
        :return: result information e.g. final score, win_status, etc.
        """

        results = {}
        protag_result = 0
        antag_result = 0
        for i, net in enumerate(actors):
            rewards = []
            wins = []
            infos = []
            kwargs = []
            for _ in range(5):
                info, states, actions, r, win, logps, entropies = rollout(net, env)
                rollout_kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps,
                                  'entropy': entropies}

                kwargs.append(rollout_kwargs)
                infos.append(info)
                rewards.append(sum(r))
                wins.append(win == 'Win')

            if i == 0:
                protag_result = rewards = np.mean(rewards)
            elif i == 1:
                antag_result = rewards = max(rewards)

            results[i] = {'info': infos,
                          'score': rewards,
                          'win': any(wins),
                          'kwargs': kwargs}

        results['adversary'] = antag_result - protag_result

        return results

    @staticmethod
    def optimize(trainer_constructor, trainer_config, registered_gym_name, level_string_monad, network_weights,
                 **kwargs):
        """Run one step of optimization!!

        :param trainer_constructor: constructor for algo to optimize wtih e.g. ppo.PPOTrainer for rllib to run optimization.
        :param trainer_config: config dict for e.g. PPO.
        :param registered_gym_name: name of env registered with ray via `env_register`
        :param level_string_monad:  callback to allow for dynamically created strings
        :param network_weights: torch state_dict
        :return: dict of {optimized weights, result_dict}
        """

        # todo same as rollout.py
        # todo will probably have to change this to first instantiate a generator model
        # and then query it for the levels.
        # trainer_config['env_config']['level_string'] = level_string_monad()
        # todo Lets define the optimize behavior here as an internal class which we pass to rllib?

        class PairedTrainingCallback(DefaultCallbacks):
            def __init__(self):
                super().__init__()

            def on_episode_start(self,
                                 *,
                                 worker: "RolloutWorker",
                                 base_env: BaseEnv,
                                 policies: Dict[PolicyID, Policy],
                                 episode: MultiAgentEpisode,
                                 env_index: Optional[int] = None,
                                 **kwargs) -> None:
                envs = base_env.get_unwrapped()
                episode.user_data['level_generation_data'] = []
                episode.user_data['generated_levels'] = []
                for e in envs:
                    episode.user_data['level_generation_data'].append(e.generation_data)
                    episode.user_data['generated_levels'].append(e.lvl)

        #     def on_episode_step(...):
        #     def on_episode_end(self,
        #                        *,
        #                        worker: "RolloutWorker",
        #                        base_env: BaseEnv,
        #                        policies: Dict[PolicyID, Policy],
        #                        episode: MultiAgentEpisode,
        #                        env_index: Optional[int] = None,
        #                        **kwargs) -> None:
        #         pass
        #         # print(episode.user_data)

            def on_postprocess_trajectory(
                    self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
                    agent_id: AgentID, policy_id: PolicyID,
                    policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
                    original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:

                adversary_dataset = episode.user_data['level_generation_data']
                print(postprocessed_batch.keys())
                # postprocessed_batch['adversary_dataset'] = adversary_dataset

            # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
            #                       **kwargs) -> None:
            #     """Called at the beginning of Policy.learn_on_batch().
            #
            #     Note: This is called before 0-padding via
            #     `pad_batch_to_sequences_of_same_size`.
            #
            #     Args:
            #         policy (Policy): Reference to the current Policy object.
            #         train_batch (SampleBatch): SampleBatch to be trained on. You can
            #             mutate this object to modify the samples generated.
            #         kwargs: Forward compatibility placeholder.
            #     """
            #
            #     pass

        #     def etc
        #
        trainer_config["callbacks"] = PairedTrainingCallback
        trainer_config['env_config']['callback_fn'] = level_string_monad
        trainer = trainer_constructor(config=trainer_config, env=registered_gym_name)
        trainer.get_policy().model.load_state_dict(network_weights)

        # todo
        # for key in network_keys:
        #     trainer.get_policy('key').model.load_state_dict(network_weights)
        result = trainer.train()

        f = {"key(s)": {'weights': 'do the same as above with keys', # trainer.get_policy(key).model.state_dict(),
                        "result_dict": result,
                        'pair_id': kwargs.get('pair_id', 0)
                        }
             }
        return f

    def get_weights(self):
        return [solver.state_dict() for solver in self.solvers]

    def set_weights(self, new_weights):
        for solver, weights in zip(self.solvers, new_weights):
            solver.load_state_dict(new_weights)



if __name__ == "__main__":
    import os
    import sys
    import ray
    import gym
    from utils.register import Registrar
    from utils.loader import load_from_yaml
    from network_factory import NetworkFactory
    from gym_factory import GridGameFactory
    from utils.gym_wrappers import SetLevelWithCallback, AlignedReward

    from tests.test_structs import example_network_factory_build_info
    import torch.optim as optim
    from generators.PCGRLGenerator import PCGRLGenerator

    from generators.RandomSelectionGenerator import RandomSelectionGenerator

    os.chdir('..')
    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=0, local_mode=True)

    args = load_from_yaml('args.yaml')
    registry = Registrar(file_args=args)

    build_info = example_network_factory_build_info
    build_info['action_space'] = gym.spaces.Discrete(169)
    build_info['num_outputs'] = 169
    build_info['name'] = 'adversary'
    build_info['model_config'] = {'length': 15, 'width': 15, "placements": 75}

    generator = PCGRLGenerator(**build_info)

    # lvls = [
    #     '''wwwwwwwwwwwww\nw....+.+++..w\nw....www....w\nw..A........w\nw...........w\nw...........w\nwwwwwww.....w\nw.g.......++w\nwwwwwwwwwwwww\n''',
    #     '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n''',
    #     '''wwwwwwwwwwwww\nw....+......w\nw........wwww\nw..A........w\nw...........w\nw.....wwwwwww\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n''',
    #     '''wwwwwwwwwwwww\nwwww.+......w\nw...........w\nw..A......eew\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    # ]
    # generator = RandomSelectionGenerator(lvls)

    nf = NetworkFactory(registry.network_name, registry.get_nn_build_info)
    gf = GridGameFactory(registry.env_name, [AlignedReward, SetLevelWithCallback])
    network = nf.make()({})

    solver = PAIREDAgentSolver([network])
    opt_dict = solver.optimize(registry.trainer_constr,
                               registry.get_trainer_config,
                               registered_gym_name=registry.name,
                               level_string_monad=generator.generate_fn_wrapper(),
                               network_weights=solver.get_weights()[0],
                               pair_id=0
                               )

    print(opt_dict)

    ray.shutdown()
