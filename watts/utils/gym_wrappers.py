import gym
import copy
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
        assert(self.play_length is not None)
        action, reward, done, info = super().step(action)
        self.steps += 1
        if "PlayerResults" in info:
            self.win = info['PlayerResults']['1']

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
        assert(isinstance(level_string, str) or level_string is None)
        kwargs['level_string'] = level_string
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def __str__(self):
        return f"<ResetCallback{str(self.env)}>"


class HierarchicalBuilderEnv(MultiAgentEnv):
    def __init__(self, env, env_config):
        MultiAgentEnv.__init__(self)
        self.game_yaml_path = env_config.get('yaml_file', None)
        self.board_shape = env_config.get("board_shape", (15, 15))
        self.egg = EnvironmentGeneratorGenerator(yaml_file=self.game_yaml_path)
        generator_yaml = self.egg.generate_env_yaml(self.board_shape)
        game = 'builder'

        build_rllib_config = {'environment_name': game,
                              'yaml_string': generator_yaml,
                              'max_steps': env_config.get('builder_max_steps', 50),
                              'global_observer_type': gd.ObserverType.ASCII,
                              'player_observer_type': gd.ObserverType.VECTOR,
                              'random_level_on_reset': False,
                              }
        self.builder_env = RLlibEnv(build_rllib_config)
        self.player_env = env

        self.action_space = gym.spaces.Dict({
            'builder': self.builder_env.action_space,
            'antagonist': self.player_env.action_space,
            'protagonist': self.player_env.action_space
        })

        self.observation_space = gym.spaces.Dict({
            'builder': self.builder_env.observation_space,
            'antagonist': self.player_env.observation_space,
            'protagonist': self.player_env.observation_space
        })

        self.builder_steps = env_config.get('builder_max_steps', 50)
        self.antagonist_steps = env_config.get('player_max_steps', 250)
        self.protagonist_steps = env_config.get('player_max_steps', 250)

        # don't record every single rollout. Instead only do every k rollouts.
        self.record_freq = env_config.get('record_freq', 10000)

        self.total_episode_steps = self.builder_steps + self.antagonist_steps + self.protagonist_steps
        self.steps_this_episode = 0

        self.final_antag = []
        self.antag_rollouts = 0
        self.sum_antag_rollout_reward = []
        self.final_protag = []
        self.protag_rollouts = 0
        self.sum_protag_rollout_reward = []
        self.final_adversary = 0
        self.num_high_level_steps = -1
        self.antagonist_agent = 'antagonist'
        self.protagonist_agent = 'protagonist'
        self.phase_counter = 0

        # both the player_env and builder_envs have the same reward_range and metadata values
        # so just get the necessary information from the player env.
        self.reward_range = self.player_env.reward_range
        self.metadata = self.player_env.metadata

        self.action_names = self.builder_env.action_names
        self.building_process = []

    @property
    def game(self):
        return self.builder_env.game

    def reset(self):
        self.cur_obs = self.builder_env.reset()
        _ = self.player_env.reset()
        self.num_high_level_steps += 1
        if self.num_high_level_steps % self.record_freq == 0:
            print("recording next rollout")
        self.steps_this_episode = 0
        self.phase_counter = 0
        self.protag_rollouts = 0
        self.antag_rollouts = 0
        self.final_antag = []
        self.final_protag = []
        self.sum_protag_rollout_reward = []
        self.sum_antag_rollout_reward = []
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.protagonist_agent_id = "{}_{}".format(
            self.protagonist_agent, self.num_high_level_steps)
        self.antagonist_agent_id = "{}_{}".format(
            self.antagonist_agent, self.num_high_level_steps)

        self.building_process.clear()

        return {
            "builder": self.cur_obs,
        }

    def step(self, action_dict):
        done = {"__all__": False}
        obs = {}
        reward = {}
        info = {}

        # build new level steps [0, 50]
        # if self.steps_this_episode < self.builder_steps:
        if self.phase_counter == 0:
            next_state, step_reward, builder_done, builder_info = self.builder_env.step(action_dict["builder"])
            if self.num_high_level_steps % self.record_freq == 0:
                lvl = self.builder_env.render(observer='global')
                self.building_process.append((self.player_env.reset(level_string=lvl, global_observations=True)['global'] / 255).astype(float))
            obs['builder'] = next_state
            reward['builder'] = step_reward
            info['builder'] = builder_info
            self.final_adversary += step_reward
            done['builder'] = builder_done

            # if final build step, move on to playing the level
            # if self.steps_this_episode == (self.builder_steps - 1):
            if builder_done:
                self.phase_counter += 1
                self.lvl = self.builder_env.render(observer='global')
                obs_dict = self.player_env.reset(level_string=self.lvl, global_observations=True)
                if self.num_high_level_steps % self.record_freq == 0:
                    self.global_view = self.building_process
                else:
                    self.global_view = None
                obs[self.antagonist_agent_id] = obs_dict['player']
                reward[self.antagonist_agent_id] = 0
                done['builder'] = True

        # run antagonist agent steps [50, 300)
        # elif self.steps_this_episode < (self.antagonist_steps + self.builder_steps):
        elif self.phase_counter == 1 and self.antag_rollouts < 1:
            # perform the step for this agent
            # ...
            next_state, step_reward, player_done, player_info = self.player_env.step(
                action_dict[self.antagonist_agent_id])
            reward[self.antagonist_agent_id] = step_reward
            info[self.antagonist_agent_id] = player_info
            self.final_antag.append(step_reward)
            if player_done:
                self.antag_rollouts += 1
                next_state = self.player_env.reset(level_string=self.lvl)
                self.sum_antag_rollout_reward.append(sum(self.final_antag))
                self.final_antag.clear()
            obs[self.antagonist_agent_id] = next_state

            # if the step that just occurred was final, then move on to next agent
            # if self.steps_this_episode == (self.antagonist_steps + self.builder_steps - 1):
            if player_done:
                _obs = self.player_env.reset(level_string=self.lvl)
                obs[self.protagonist_agent_id] = _obs
                self.phase_counter += 1
                done[self.antagonist_agent_id] = True

        # run protagonist agent steps [300, 550]
        # elif self.steps_this_episode < (self.antagonist_steps + self.builder_steps + self.protagonist_steps):
        elif self.phase_counter == 2 and self.protag_rollouts < 1:
            # perform the step for this agent
            # ...
            next_state, step_reward, player_done, player_info = self.player_env.step(
                action_dict[self.protagonist_agent_id])
            reward[self.protagonist_agent_id] = step_reward
            info[self.protagonist_agent_id] = player_info
            self.final_protag.append(step_reward)
            if player_done:
                self.protag_rollouts += 1
                next_state = self.player_env.reset(level_string=self.lvl)
                self.sum_protag_rollout_reward.append(sum(self.final_protag))
                self.final_protag.clear()
            obs[self.protagonist_agent_id] = next_state

            if player_done:
                self.phase_counter += 1
                done[self.protagonist_agent_id] = True
                done["__all__"] = True

        self.steps_this_episode += 1
        return obs, reward, done, info

    def close(self):
        self.player_env.close()
        self.builder_env.close()

    def __str__(self):
        return f"<HierarchicalBuilder{str(self.player_env)}>"


class Regret(HierarchicalBuilderEnv):
    def __init__(self, env, env_config, passthrough_reward=False):
        HierarchicalBuilderEnv.__init__(self, env, env_config)
        self.regret = 0
        self.passthrough_reward = passthrough_reward

    def step(self, action_dict):
        ns, rew, d, info = super().step(action_dict)
        # if episode is over, calculate regret
        if d['__all__']:
            # R = max(antagonist) - mean(protagonist)
            self.regret = max(self.sum_antag_rollout_reward) - (
                        sum(self.sum_protag_rollout_reward) / len(self.sum_protag_rollout_reward))
            # set builder to +reward
            # set antagonist to +reward
            # set protagonist to -reward
            # send these values to rllib for them to optimize it
            for agent in [self.antagonist_agent_id, self.protagonist_agent_id, "builder"]:
                if agent == self.antagonist_agent_id or agent == 'builder':
                    rew[agent] = self.regret
                else:
                    rew[agent] = -self.regret
        # zero out the reward that's not regret
        elif not self.passthrough_reward:
            for k, v in rew.items():
                rew[k] = 0
        return ns, rew, d, info

    def reset(self):
        return super().reset()

    def __str__(self):
        return f"<Regret{str(self.player_env)}>"


class PlacePredefinedSequence(gym.Wrapper, MultiAgentEnv):
    """

    """
    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self)
        assert isinstance(env, HierarchicalBuilderEnv)
        self.action_names = self.env.action_names
        self.n_builder_steps = self.env.builder_steps + 1
        self.board_shape = self.env.board_shape

        # update the action space that rllib will see so that rllib doesn't know about
        #  the `object` and `place/not-place` dimensions.
        #  Instead, rllib will only know about the [x, y] dimensions of the MultiDiscrete action spaces
        #  This wrapper assumes direct control of what objects to place and in what sequence.
        #  This will allow generation of the HierarchicalBuilderEnv to function on any griddly game and
        #   function like how Dennis et al placed objects in https://arxiv.org/abs/2012.02096
        self.action_space = gym.spaces.Dict({
            'builder': gym.spaces.MultiDiscrete(list(self.board_shape)),
            'protagonist': self.env.player_env.action_space,
            'antagonist': self.env.player_env.action_space
        })

        self.observation_space = self.env.observation_space
        self.placement_counter = 0
        self.locs = []
        self.user_seq = env_config.get('user_seq', None)
        # For now, we're assuming that wrapper is being used on the maze domain.
        if self.user_seq is None:
            self.user_seq = ['place_avatar', 'place_exit']
            for _ in range(self.n_builder_steps):
                self.user_seq.append('place_wall')

    def step(self, action_dict):
        if 'builder' in action_dict.keys():
            x, y = action_dict['builder']
            self.locs.append((x, y))
            a = self.user_seq[self.placement_counter]
            # if we tried to place the exit in the same place as the avatar, place the exit randomly.
            if a == 'place_exit' and (x, y) == self.locs[0]:
                x, y = self.action_space['builder'].sample()
            action_dict['builder'] = [x, y, self.action_names.index(a), 1]
            self.placement_counter += 1
        ns, rew, d, info = super().step(action_dict)
        return ns, rew, d, info

    def reset(self, **kwargs):
        self.placement_counter = 0
        self.locs.clear()
        return super().reset(**kwargs)

    def __str__(self):
        return f"<Seq{str(self.env)}>"


def add_wrappers(str_list: list) -> list:
    wraps = []
    for w in str_list:
        if "Aligned" in w:
            wraps.append(AlignedReward)
        elif "ResetCallback" in w:
            wraps.append(SetLevelWithCallback)
        # This is useful; see the griddly multiagent rllib interface doc page
        # https://griddly.readthedocs.io/en/latest/rllib/multi-agent/index.html
        elif "MultiAgent" in w:
            wraps.append(RLlibMultiAgentWrapper)
        elif "HierarchicalBuilder" in w:
            wraps.append(HierarchicalBuilderEnv)
        elif 'Regret' in w:
            wraps.append(Regret)
        else:
            raise ValueError("Requested wrapper does not exist. Please make it.")

    return wraps


if __name__ == "__main__":
    import gym
    from griddly import gd
    import os
    import ray
    import matplotlib.pyplot as plt

    while 'paired.py' not in os.listdir('.'):
        os.chdir('..')

    env_config = {
        'yaml_file': os.path.join('example_levels', 'endless_maze.yaml'),
        'board_shape': (15, 15),
        'global_observer_type': gd.ObserverType.BLOCK_2D
    }

    env = RLlibEnv(env_config)
    env = Regret(env, env_config)
    env = PlacePredefinedSequence(env, env_config)
    s = env.reset()

    done = False
    ims = []
    while not done:
        ns, r, d, i = env.step({'builder': env.action_space['builder'].sample()})
        # ims.append((i['builder']['image'] / 255).astype(float))
        done = d['builder']

    # for im in ims:
    #     plt.imshow(im)
    #     plt.show()
