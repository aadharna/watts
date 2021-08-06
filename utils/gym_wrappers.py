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
        assert(self.play_length is not None)
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
        info['win']  = self.win
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
    def __init__(self, env, env_config, game_yaml_path, board_shape):
        super().__init__(env, env_config)
        self.egg = EnvironmentGeneratorGenerator(yaml_file=game_yaml_path)
        self.board_shape = board_shape
        generator_yaml = self.egg.generate_env_yaml(self.board_shape)
        game = 'builder'

        build_rllib_config = {'environment_name': game,
                              'yaml_string': generator_yaml,
                              'max_steps': env_config.get('max_steps', 50),
                              'global_observer_type': gd.ObserverType.ASCII,
                              'player_observer_type': gd.ObserverType.ASCII,
                              'random_level_on_reset': False,
                              }
        self.builder_env = RLlibEnv(build_rllib_config)
        self.player_env = env

    def reset(self):
        self.cur_obs = self.flat_env.reset()
        self.current_goal = None
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.low_level_agent_id = "low_level_{}".format(
            self.num_high_level_steps)
        return {
            "builder_agent": self.cur_obs,
        }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "builder_agent" in action_dict:
            return self._high_level_step(action_dict["builder_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        # logger.debug("High level agent sets goal")
        self.current_goal = action
        self.steps_remaining_at_level = 250
        self.num_high_level_steps += 1
        self.low_level_agent_id = "low_level_{}".format(
            self.num_high_level_steps)
        obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        # logger.debug("Low level agent step {}".format(action))
        self.steps_remaining_at_level -= 1
        cur_pos = tuple(self.cur_obs[0])
        goal_pos = self.flat_env._get_new_pos(cur_pos, self.current_goal)

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        new_pos = tuple(f_obs[0])
        self.cur_obs = f_obs

        # Calculate low-level agent observation and reward
        obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
        if new_pos != cur_pos:
            if new_pos == goal_pos:
                rew = {self.low_level_agent_id: 1}
            else:
                rew = {self.low_level_agent_id: -1}
        else:
            rew = {self.low_level_agent_id: 0}

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            # logger.debug("high level final reward {}".format(f_rew))
            rew["builder_agent"] = f_rew
            obs["builder_agent"] = f_obs
        elif self.steps_remaining_at_level == 0:
            done[self.low_level_agent_id] = True
            rew["builder_agent"] = 0
            obs["builder_agent"] = f_obs

        return obs, rew, done, {}



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
    os.chdir('..')

    registery = Registrar(file_args=load_from_yaml('args.yaml'))

    env = RLlibEnv(registery.get_config_to_build_rllib_env)
    print(str(env))
    env = AlignedReward(env, registery.get_config_to_build_rllib_env)
    print(str(env))


    done = False
    rs = []
    infos = []
    while not done:
        ns, r, done, i = env.step(env.action_space.sample())
        rs.append(r)
        infos.append(i)

    print(rs)
    # print(infos)
