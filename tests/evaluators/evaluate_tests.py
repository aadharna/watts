import os
import logging
import copy
import pytest

from watts import gym_factory
from watts import network_factory
from watts.evaluators.rollout import rollout

import tests.test_structs as test_structs


def test_evaluate():
    gf = gym_factory.GridGameFactory("foo", [])
    yaml_file = os.path.join('example_levels', 'limited_zelda.yaml')
    rllib_env_config = {
        'yaml_file': yaml_file,
        'level_string': 'wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'
    }
    env = gf.make()(rllib_env_config)

    build_info = copy.copy(test_structs.example_network_factory_build_info)
    build_info['name'] = network_factory.aiide
    nf = network_factory.NetworkFactory(network_factory.aiide, build_info)
    actor = nf.make()(test_structs.example_aiide_state_dict)

    info, states, actions, rewards, win, logps, entropies = rollout(actor, env, 'cpu')
    logging.info(info)
    logging.info(rewards)
    logging.info(sum(rewards))
    logging.info(win)
