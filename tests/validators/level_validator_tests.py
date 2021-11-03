import sys
sys.path.append('..')
import os
import ray
import unittest
import sys
sys.path.append('../../')
import network_factory
from generators.PCGRLGenerator import PCGRLGenerator
from generators.static_generator import StaticGenerator
from tests.test_structs import example_pcgrl_network_factory_build_info, example_network_factory_build_info
from tests.test_classes import MockSolver
from validators.agent_validator import RandomAgentValidator
from validators.graph_validator import GraphValidator


class TestLevelValidators(unittest.TestCase):

    def test_graph_validator_on_PCGRL(self):
        build_info = example_pcgrl_network_factory_build_info
        build_info['name'] = 'Adversarial_PCGRL'
        generator = PCGRLGenerator(**build_info)
        validator = GraphValidator()
        res = validator.validate_level([generator], [MockSolver()])
        print(res)

    def test_graph_validator_on_static(self):
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        generator = StaticGenerator(level_string=level_string)
        validator = GraphValidator()
        solver = MockSolver()
        assert (validator.validate_level([generator], [solver]))

    def test_random_agent_validator_on_static(self):
        build_info = example_network_factory_build_info
<<<<<<< HEAD
        build_info['network_name'] = 'SimpleConvAgent'
=======
        build_info['network_name'] = 'AIIDE_PINSKY_MODEL'
>>>>>>> 42fac187c9ee4d1ca342e447497409ad08841bce
        level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
        nf = network_factory.NetworkFactory(build_info)
        generator = StaticGenerator(level_string=level_string)
        validator = RandomAgentValidator(network_factory_monad=nf.make(),
                                         env_config={'yaml_file': os.path.join('levels', 'limited_zelda.yaml')})
        # the mock class isn't a real ray.remote-ified class
        # therefore we cannot call ray.get on the output from `get_weights`
        # with self.assertRaises(ValueError):
<<<<<<< HEAD
        #     res = validator.validate_level([generator], [MockSolver()])
=======
        #     res = validator.validate_level(generator, MockSolver())
>>>>>>> 42fac187c9ee4d1ca342e447497409ad08841bce

if __name__ == '__main__':
    unittest.main()

