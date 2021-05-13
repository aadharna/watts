from griddly.util.rllib.environment.core import RLlibEnv
import gym
import gym_factory
import unittest


class TestGymFactory(unittest.TestCase):

    def test_simple(self):
        gf = gym_factory.GridGameFactory("foo", [])
        g = gf.make()({'yaml_file': '../levels/limited_zelda.yaml'})
        assert g._enable_history

    def test_with_conf(self):
        gf = gym_factory.GridGameFactory("foo", [])
        g = gf.make()({'yaml_file': '../levels/limited_zelda.yaml', 'generate_valid_action_trees': True})
        assert g._enable_history
        assert g.generate_valid_action_trees

    def test_with_wrapper(self):
        gf = gym_factory.GridGameFactory("foo", [SimpleGymWrapper])
        g = gf.make()({'yaml_file': '../levels/limited_zelda.yaml'})
        assert g.foo == 5


class SimpleGymWrapper(gym.Wrapper, RLlibEnv):
    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env=env)
        RLlibEnv.__init__(self, env_config=env_config)

        self.foo = 5
