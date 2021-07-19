from generators.PCGRLGenerator import PCGRLGenerator
from generators.AIIDE_generator import EvolutionaryGenerator
from generators.static_generator import StaticGenerator
from generators.RandomSelectionGenerator import RandomSelectionGenerator

from mutation.level_validator import RandomAgentValidator, GraphValidator

import gym_factory

from tests.test_structs import example_network_factory_build_info
import gym

import os
import unittest

class TestLevelValidators(unittest.TestCase):

    def test_graph_validator_on_PCGRL(self):
        build_info = example_network_factory_build_info
        build_info['action_space'] = gym.spaces.Discrete(169)
        build_info['num_outputs'] = 169
        build_info['name'] = 'adversary'
        build_info['model_config'] = {'length': 15, 'width': 15, "placements": 75}

        generator = PCGRLGenerator(**build_info)
        validator = GraphValidator()
        assert(validator.validate_level(generator))

    def test_graph_validator_on_static(self):
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        generator = StaticGenerator(level_string=level_string)
        validator = GraphValidator()
        assert (validator.validate_level(generator))

    def test_random_agent_validator_on_static(self):
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        gf = gym_factory.GridGameFactory("foo", [])
        generator = StaticGenerator(level_string=level_string)
        validator = RandomAgentValidator()
        assert (validator.validate_level(generator,
                                         gym_factory_monad=gf.make(),
                                         config={'yaml_file': os.path.join('levels', 'limited_zelda.yaml')}))
