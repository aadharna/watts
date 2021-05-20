import copy
from evaluators.evaluate import evaluate
import network_factory
import gym_factory
import tests.test_structs as test_structs
import unittest


class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        gf = gym_factory.GridGameFactory("foo", [])
        rllib_env_config = {
            'yaml_file': 'levels/limited_zelda.yaml',
            'level_string': 'wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'
        }
        env = gf.make()(rllib_env_config)

        build_info = copy.copy(test_structs.example_network_factory_build_info)
        build_info['name'] = network_factory.aiide
        nf = network_factory.NetworkFactory(network_factory.aiide, build_info)
        actor = nf.make()(test_structs.example_aiide_state_dict)

        info, rewards, win = evaluate(actor, env)
        print(info)
        print(rewards)
        print(sum(rewards))
        print(win)