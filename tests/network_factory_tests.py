import sys
sys.path.append('..')
import copy
from typing import Callable
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent
from models.AIIDE_network import AIIDEActor
from models.PCGRL_network import PCGRLAdversarial
from network_factory import NetworkFactory
from tests import test_structs
import unittest


def loadConfigs():
    
    networks = ['AIIDE_PINSKY_MODEL', 'SimpleConvAgent', 'GAPAgent', 'Adversarial_PCGRL']
    build_configs = {}
    for i, network in enumerate(networks):
        config = {}
        if network == 'Adversarial_PCGRL':
            config = copy.copy(test_structs.example_pcgrl_network_factory_build_info)
        else:
            config = copy.copy(test_structs.example_network_factory_build_info)

        config['network_name'] = network
        build_configs[network] = config

    return build_configs


CONFIGS = loadConfigs()

class TestNetworkFactory(unittest.TestCase):


    def test_aiids(self):
        factory = NetworkFactory(CONFIGS['AIIDE_PINSKY_MODEL']).make()
        assert isinstance(factory, Callable)
        assert(factory().__class__.__name__ == 'AIIDEActor')

    def test_conv(self):
        factory = NetworkFactory(CONFIGS['SimpleConvAgent']).make()
        assert isinstance(factory, Callable)
        assert(factory().__class__.__name__ == 'SimpleConvAgent')

    def test_gap(self):
        factory = NetworkFactory(CONFIGS['GAPAgent']).make()
        assert isinstance(factory, Callable)

    def test_pcgrl(self):
        factory = NetworkFactory(CONFIGS['Adversarial_PCGRL']).make()
        assert isinstance(factory, Callable)
        assert(factory().__class__.__name__ == 'PCGRLAdversarial')

    def test_multiple_policies(self):
        agents = []
        for i, (_, config) in enumerate(copy.copy(CONFIGS).items()):
            config['agent'] = f'agent-{i}'
            agents.append(config)
        
        factory = NetworkFactory(agents).make()
        # return type for multi-agent config should be a dictionary
        assert isinstance(factory, dict)

        # make sure that the keys in the dictionary match up with the agent IDs
        for i, agent_factory in enumerate(factory):
            assert agent_factory == f'agent-{i}'



if __name__ == '__main__':
    unittest.main()

