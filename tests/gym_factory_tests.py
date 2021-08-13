import gym_factory
import unittest
from tests.test_classes import SimpleGymWrapper


class TestGymFactory(unittest.TestCase):

    def test_simple(self):
        gf = gym_factory.GridGameFactory("foo", [])
        g = gf.make()({'yaml_file': 'levels/limited_zelda.yaml'})
        assert g._enable_history

    def test_with_conf(self):
        gf = gym_factory.GridGameFactory("foo", [])
        g = gf.make()({'yaml_file': 'levels/limited_zelda.yaml', 'generate_valid_action_trees': True})
        assert g._enable_history
        assert g.generate_valid_action_trees

    def test_with_wrapper(self):
        gf = gym_factory.GridGameFactory("foo", [SimpleGymWrapper])
        g = gf.make()({'yaml_file': 'levels/limited_zelda.yaml'})
        assert g.foo == 5



