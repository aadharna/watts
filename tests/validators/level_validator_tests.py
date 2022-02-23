import os
import ray
import numpy as np
import unittest

from watts import network_factory
from watts.generators.PCGRLGenerator import PCGRLGenerator
from watts.generators.StaticGenerator import StaticGenerator
from watts.validators.agent_validator import RandomAgentValidator
from watts.validators.graph_validator import GraphValidator

from tests.test_structs import \
        example_pcgrl_network_factory_build_info, \
        example_network_factory_build_info
from tests.test_classes import MockSolver


class MockGameSchema:
    def __init__(self):
        self.agent_chars = {'A'}
        self.wall_char = 'w'
        self.interesting_chars = {'+', 'e', 'g'}


class TestLevelValidators(unittest.TestCase):

    # def test_graph_validator_on_PCGRL(self):
    #     build_info = example_pcgrl_network_factory_build_info
    #     build_info['name'] = 'pcgrl_test'
    #     generator = PCGRLGenerator(**build_info)
    #     validator = GraphValidator(MockGameSchema())
    #     res = validator.validate_level([generator], [MockSolver()])
    #     print(res)

    def test_graph_validator_on_static(self):
        level_string = '''w w w w w w w w w w w w w\nw . . . . + e . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n'''
        generator = StaticGenerator(level_string=level_string)
        validator = GraphValidator(MockGameSchema())
        solver = MockSolver()
        result, data = validator.validate_level([generator], [solver])
        assert result

    def test_graph_validator_on_static_failure(self):
        level_string = '''w w w w w\nw A w . w\nw .         w . w\nw + w g w\nw w w w w\n'''
        generator = StaticGenerator(level_string=level_string)
        validator = GraphValidator(MockGameSchema())
        solver = MockSolver()
        result, data = validator.validate_level([generator], [solver])
        assert not result

    def test_random_agent_validator_on_static(self):
        build_info = example_network_factory_build_info
        build_info['name'] = 'conv_test'
        # the mock class isn't a real ray.remote-ified class
        # therefore we cannot call ray.get on the output from `get_weights`
        # level_string = '''w w w w w w w w w w w w w\nw . . . . + e . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n'''
        # nf = network_factory.NetworkFactory(network_name=network_factory.conv, nn_build_info=build_info)
        # generator = StaticGenerator(level_string=level_string)
        # validator = RandomAgentValidator(network_factory_monad=nf.make(),
        #                                  env_config={'yaml_file': os.path.join('example_levels', 'limited_zelda.yaml')},
        #                                  high_cutoff=np.inf,
        #                                  low_cutoff=-np.inf)
        # with self.assertRaises(ValueError):
        #     res = validator.validate_level([generator], [MockSolver()])
