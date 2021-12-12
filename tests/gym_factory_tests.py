import os
import unittest
from watts import gym_factory
from tests.test_classes import SimpleGymWrapper


class TestGymFactory(unittest.TestCase):

    def test_simple(self):
        gf = gym_factory.GridGameFactory("foo", [])
        yaml_file = os.path.join('example_levels', 'limited_zelda.yaml')
        g = gf.make()({'yaml_file': yaml_file})
        assert g._enable_history

    def test_with_conf(self):
        gf = gym_factory.GridGameFactory("foo", [])
        yaml_file = os.path.join('example_levels', 'limited_zelda.yaml')
        g = gf.make()({
                'yaml_file': yaml_file,
                'generate_valid_action_trees': True
            })
        assert g._enable_history
        assert g.generate_valid_action_trees

    def test_with_wrapper(self):
        gf = gym_factory.GridGameFactory("foo", [SimpleGymWrapper])
        yaml_file = os.path.join('example_levels', 'limited_zelda.yaml')
        g = gf.make()({'yaml_file': yaml_file})
        assert g.foo == 5



