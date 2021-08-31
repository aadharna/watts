import gym
import os
import unittest

from generators.PCGRLGenerator import PCGRLGenerator
from generators.static_generator import StaticGenerator
import gym_factory
from tests.test_structs import example_pcgrl_network_factory_build_info
from evolution.level_validator import RandomAgentValidator, GraphValidator


class TestLevelValidators(unittest.TestCase):

    def test_graph_validator_on_PCGRL(self):
        build_info = example_pcgrl_network_factory_build_info
        build_info['name'] = 'pcgrl_test'
        generator = PCGRLGenerator(**build_info)
        validator = GraphValidator()
        res = validator.validate_level(generator)
        print(res)

    def test_graph_validator_on_static(self):
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        generator = StaticGenerator(level_string=level_string)
        validator = GraphValidator()
        assert (validator.validate_level(generator))

    def test_random_agent_validator_on_static(self):
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        gf = gym_factory.GridGameFactory("foo", [])
        generator = StaticGenerator(level_string=level_string)
        validator = RandomAgentValidator(gym_factory_monad=gf.make(), env_config={'yaml_file': os.path.join('levels', 'limited_zelda.yaml')})
        res = validator.validate_level(generator)
        print(res)
