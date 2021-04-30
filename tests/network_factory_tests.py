from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent
from gym.spaces import Box, MultiDiscrete
from models.AIIDE_network import AIIDEActor
from models.PCGRL_networks import PCGRLAdversarial
import numpy as np
import network_factory
import unittest


def run_network_factory_test(name: str, constructor):
    build_info = {
        'action_space': MultiDiscrete([5, 2, 2]),
        'obs_space': Box(0.0, 255.0, (5, 5, 6), np.float64),
        'model_config': {},
        'num_outputs': 7,
        'name': name
    }
    nf = network_factory.NetworkFactory(name, build_info)
    assert nf.network_name == name
    assert nf.nn_build_info == build_info
    assert nf.constructor == constructor
    # Confirm actually making a network doesn't break things
    network = nf.make()()
    assert network is not None


class TestNetworkFactory(unittest.TestCase):

    def test_aiide(self):
        run_network_factory_test(network_factory.aiide, AIIDEActor)

    def test_conv(self):
        run_network_factory_test(network_factory.conv, SimpleConvAgent)

    def test_gap(self):
        run_network_factory_test(network_factory.gap, GAPAgent)

    def test_pcgrl(self):
        run_network_factory_test(network_factory.pcgrl, PCGRLAdversarial)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            network_factory.NetworkFactory("foo", {})