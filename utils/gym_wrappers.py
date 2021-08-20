import gym
from griddly import gd
from griddly.util.environment_generator_generator import EnvironmentGeneratorGenerator
from griddly.util.rllib.environment.core import RLlibEnv, RLlibMultiAgentWrapper
from ray.rllib.env import MultiAgentEnv


class AlignedReward(gym.Wrapper, RLlibEnv):

    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env=env)
        RLlibEnv.__init__(self, env_config=env_config)

        self.env = env
        _ = self.env.reset()
        self.env.enable_history(True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.win = None
        self.steps = -1
        self.play_length = env_config.get('max_steps', 500)

    def step(self, action):
        assert (self.play_length is not None)
        action, reward, done, info = super().step(action)
        self.steps += 1
        if "PlayerResults" in info:
            self.win = info['PlayerResults']['1']
            # print(f"set win to: {self.win}")

        if self.win == 'Win':
            reward = 1 - (self.steps / self.play_length)
        elif self.win == 'Lose':
            reward = (self.steps / self.play_length) - 1
        else:
            reward = 0
        info['step'] = self.steps
        info['win'] = self.win
        info['r'] = reward

        return action, reward, done, info

    def reset(self, **kwargs):
        self.win = None
        self.steps = -1
        return self.env.reset(**kwargs)

    def __str__(self):
        return f"<Aligned{str(self.env)}>"


class SetLevelWithCallback(gym.Wrapper):
    """GymWrapper to set the level with a callback function in Griddly.

    The callback_fn should output a string representation of the level.
    """

    def __init__(self, env, env_config):
        super(SetLevelWithCallback, self).__init__(env=env)
        self.create_level_fn = env_config.get('callback_fn', lambda: (None, None))
        self.generation_data = None
        self.lvl = None

    def reset(self, **kwargs):
        level_string, info_dict = self.create_level_fn()
        self.lvl = level_string
        self.generation_data = info_dict
        assert (isinstance(level_string, str) or level_string is None)
        kwargs['level_string'] = level_string
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def __str__(self):
        return f"<ResetCallback{str(self.env)}>"



class HierarchicalBuilderEnv(MultiAgentEnv):
    def __init__(self, env, env_config):
        super().__init__()
        self.game_yaml_path = env_config.get('yaml_file', None)
        self.board_shape = env_config.get("board_shape", (15, 15))
        self.egg = EnvironmentGeneratorGenerator(yaml_file=self.game_yaml_path)
        generator_yaml = self.egg.generate_env_yaml(self.board_shape)
        game = 'builder'

        build_rllib_config = {'environment_name': game,
                              'yaml_string': generator_yaml,
                              'max_steps': env_config.get('builder_max_steps', 50),
                              'global_observer_type': gd.ObserverType.ASCII,
                              'player_observer_type': gd.ObserverType.ASCII,
                              'random_level_on_reset': False,
                              }
        self.builder_env = RLlibEnv(build_rllib_config)
        self.player_env = env

        self.builder_steps = env_config.get('builder_max_steps', 50)
        self.antagonist_steps = env_config.get('max_steps', 250)
        self.protagonist_steps = env_config.get('max_steps', 250)
        self.total_episode_steps = self.builder_steps + self.antagonist_steps + self.protagonist_steps
        self.steps_this_episode = 0

        self.final_antag = []
        self.antag_rollouts = 0
        self.final_protag = []
        self.protag_rollouts = 0
        self.final_adversary = 0
        self.num_high_level_steps = 0
        self.antagonist_agent = 'antagonist'
        self.protagonist_agent = 'protagonist'
        self.phase_counter = 0


    def reset(self):
        self.cur_obs = self.builder_env.reset()
        _ = self.player_env.reset()
        self.num_high_level_steps += 1
        self.steps_this_episode = 0
        self.phase_counter = 0
        self.protag_rollouts = 0
        self.antag_rollouts = 0
        self.final_antag = []
        self.final_protag = []
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.protagonist_agent_id = "{}_{}".format(
            self.protagonist_agent, self.num_high_level_steps)
        self.antagonist_agent_id = "{}_{}".format(
            self.antagonist_agent, self.num_high_level_steps)

        return {
            "builder": self.cur_obs,
        }

    def step(self, action_dict):
        # assert len(action_dict) == 1, action_dict
        done = {"__all__": False}
        obs = {}
        reward = {}
        info = {}
        print(action_dict.keys())

        # build new level steps [0, 50)
        # if self.steps_this_episode < self.builder_steps:
        if self.phase_counter == 0:
            # perform singular build step
            next_state, step_reward, builder_done, builder_info = self.builder_env.step(action_dict["builder"])
            obs['builder'] = next_state
            reward['builder'] = step_reward
            info['builder'] = builder_info
            self.final_adversary += step_reward
            done['builder'] = builder_done
            # info['who'] = 'builder'

            # if final build step, move on to playing the level
            # if self.steps_this_episode == (self.builder_steps - 1):
            if builder_done:
                self.phase_counter += 1
                lvl = self.builder_env.render(observer='global')
                _obs = self.player_env.reset(level_string=lvl)
                obs[self.antagonist_agent_id] = _obs
                reward[self.antagonist_agent_id] = 0
                # info['who'] = self.antagonist_agent_id

        # run antagonist agent steps [50, 300)
        # elif self.steps_this_episode < (self.antagonist_steps + self.builder_steps):
        elif self.phase_counter == 1 and self.antag_rollouts < 5:
            # perform the step for this agent
            # ...
            next_state, step_reward, player_done, player_info = self.player_env.step(
                action_dict[self.antagonist_agent_id])
            if player_done:
                self.antag_rollouts += 1
                lvl = self.builder_env.render(observer='global')
                next_state = self.player_env.reset(level_string=lvl)
            obs[self.antagonist_agent_id] = next_state
            reward[self.antagonist_agent_id] = step_reward
            info[self.antagonist_agent_id] = player_info
            self.final_antag.append(step_reward)
            # info['who'] = self.antagonist_agent_id

            # if the step that just occurred was final, then move on to next agent
            # if self.steps_this_episode == (self.antagonist_steps + self.builder_steps - 1):
            if player_done and self.antag_rollouts == 5:
                lvl = self.builder_env.render(observer='global')
                _obs = self.player_env.reset(level_string=lvl)
                obs[self.protagonist_agent_id] = _obs
                reward[self.protagonist_agent_id] = 0
                # info['who'] = self.protagonist_agent_id
                self.phase_counter += 1
                done[self.antagonist_agent_id] = True

        # run protagonist agent steps [300, 550]
        # elif self.steps_this_episode < (self.antagonist_steps + self.builder_steps + self.protagonist_steps):
        elif self.phase_counter == 2 and self.protag_rollouts < 5:
            # perform the step for this agent
            # ...
            next_state, step_reward, player_done, player_info = self.player_env.step(
                action_dict[self.protagonist_agent_id])
            if player_done:
                self.protag_rollouts += 1
                lvl = self.builder_env.render(observer='global')
                next_state = self.player_env.reset(level_string=lvl)
            obs[self.protagonist_agent_id] = next_state
            reward[self.protagonist_agent_id] = step_reward
            info[self.protagonist_agent_id] = player_info
            self.final_protag.append(step_reward)
            # info['who'] = self.protagonist_agent_id

            if player_done and self.protag_rollouts == 5:
                # info['who'] = self.protagonist_agent_id
                self.phase_counter += 1
                done[self.protagonist_agent_id] = True

        if self.phase_counter == 3:
            done["__all__"] = True
            # info['who'] = 'builder'
            self.regret = max(self.final_antag) - (sum(self.final_protag) / self.protag_rollouts)
            for agent in [self.antagonist_agent_id, self.protagonist_agent_id, "builder"]:
                if agent == self.antagonist_agent_id or agent == 'builder':
                    reward[agent] = self.regret
                else:
                    reward[agent] = -self.regret

        self.steps_this_episode += 1
        return obs, reward, done, info


def add_wrappers(str_list: list) -> list:
    wraps = []
    for w in str_list:
        if "Aligned" in w:
            wraps.append(AlignedReward)
        elif "ResetCallback" in w:
            wraps.append(SetLevelWithCallback)
        elif "MultiAgent" in w:
            wraps.append(RLlibMultiAgentWrapper)
        else:
            raise ValueError("Requested wrapper does not exist. Please make it.")

    return wraps


if __name__ == "__main__":
    import gym
    import os
    from griddly.util.rllib.environment.core import RLlibEnv
    from utils.register import Registrar
    from utils.loader import load_from_yaml
    import ray
    from ray import tune
    from ray.tune import function
    from ray.tune.registry import register_env
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.models import ModelCatalog
    # from ray.rllib.examples.self_play_with_open_spiel_connect_4 import config as spiel_config
    # from ray.rllib.examples.hierarchical_training import
    import sys

    from models.AIIDE_network import AIIDEActor
    from models.PCGRL_network import PCGRLAdversarial

    os.chdir('..')

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1, ignore_reinit_error=True)  # , log_to_driver=False, local_mode=True)

    registery = Registrar(file_args=load_from_yaml('args.yaml'))

    config = registery.get_config_to_build_rllib_env
    config['board_shape'] = (15, 15)
    config['builder_max_steps'] = 50
    config['max_steps'] = 250


    def make_env(config):
        print(config['yaml_file'])
        env = RLlibEnv(config)
        env = AlignedReward(env, config)
        h_env = HierarchicalBuilderEnv(env, config)
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
    register_env('h_zelda', make_env)

    from ray.rllib.agents.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
    from ray.rllib.policy import Policy
    from ray.rllib.utils.typing import AgentID, PolicyID
    from ray.rllib.policy.sample_batch import SampleBatch
    from typing import Dict, Optional

    # class PairedTrainingCallback(DefaultCallbacks):
    #     def __init__(self):
    #         super().__init__()
    #
    #     def on_postprocess_trajectory(
    #             self, *,
    #             worker: "RolloutWorker",
    #             episode: MultiAgentEpisode,
    #             agent_id: AgentID,
    #             policy_id: PolicyID,
    #             policies: Dict[PolicyID, Policy],
    #             postprocessed_batch: SampleBatch,
    #             original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
    #         print(episode)
    #         print(agent_id, policy_id)
    #         print(original_batches.keys())
    #         print(original_batches.values())
    #         print(postprocessed_batch.keys())


    h_env = make_env(config)
    # print(h_env.builder_env.action_space)
    _ = h_env.reset()
    config2 = {
        'env': 'h_zelda',
        'num_workers': 1,
        "num_envs_per_worker": 2,
        'env_config': config,
        # "callbacks": PairedTrainingCallback,
        'multiagent': {
            'policies': {
                'builder': (None, h_env.builder_env.observation_space,
                            h_env.builder_env.action_space, {'model': {'custom_model': 'PCGRL',
                                                                       'custom_model_config': {'cell_size': 2704}}}),
                'antagonist': (None, h_env.player_env.observation_space,
                               h_env.player_env.action_space, {'model': {'custom_model': 'AIIDE',
                                                                         'custom_model_config': {}}}),
                'protagonist': (None, h_env.player_env.observation_space,
                                h_env.player_env.action_space, {'model': {'custom_model': 'AIIDE',
                                                                          'custom_model_config': {}}})
            },
            'policy_mapping_fn': policy_mapping_fn
        },
        "framework": 'torch',
    }

    try:
        trainer = PPOTrainer(config=config2, env="h_zelda")
        result = trainer.train()
        pass
    except Exception as e:
        error = e
        print(error)
        ray.shutdown()
    ray.shutdown()

    #
    #
    #
    # rs = []
    # obs = []
    # dones = []
    # done = {'__all__': False}
    # infos = {}
    # infos['who'] = 'builder'
    # while not done['__all__']:
    #     if infos['who'] == 'builder':
    #         action = h_env.builder_env.action_space.sample()
    #     else:
    #         action = h_env.player_env.action_space.sample()
    #     action_dict = {f"{infos['who']}": action}
    #     ns, reward, done, infos = h_env.step(action_dict)
    #     rs.append(reward)
    #     obs.append(ns)
    #     dones.append(done)
