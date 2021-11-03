from ray.rllib.models import ModelCatalog

from models.AIIDE_network import AIIDEActor
from models.PCGRL_network import PCGRLAdversarial
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent

from typing import Union, Callable

aiide = "AIIDE_PINSKY_MODEL"
conv = "SimpleConvAgent"
gap = "GAPAgent"
pcgrl = "Adversarial_PCGRL"


def get_network_constructor(network_name: str):
    if network_name == aiide:
        return AIIDEActor
    elif network_name == pcgrl:
        return PCGRLAdversarial
    elif network_name == conv:
        return SimpleConvAgent
    elif network_name == gap:
        return GAPAgent
    else:
        raise ValueError("Network unavailable. Add the network definition to the models folder and network_factory")


class NetworkFactory:
    def __init__(self, network_config: Union[dict, list]):
        """
        :param dict | list[dict]: config or list of config options for each agent in the environment

        :return dict: dict of constructors for each network specified by config
        """
        

        if isinstance(network_config, dict):
            return_dict = True
        elif isinstance(network_config, list):
            return_dict = False

        self.return_dict = True
        if isinstance(network_config, dict):
            self.return_dict = False
            network_config['agent'] = 'default'
            network_config = [network_config]

        self.network_config = network_config

        self.constructors = {}
        for config in self.network_config:
            agent = config['agent']
            network_name = config['network_name']
            constructor = get_network_constructor(network_name)
            ModelCatalog.register_custom_model(network_name, constructor)
            config['constructor'] = constructor
    

    def make(self):
        """
        Returns make functions used to build network
        If there are multiple agent configs passed as a list, this
        function will return a dictionary where each key is the network name
        and each corresponding value is the network's make function
        """
        def _make(config):
            def __make(state_dict=None):
                constructor = config['constructor']
                network = constructor(
                            config['obs_space'],
                            config['action_space'],
                            config['num_outputs'],
                            config['model_config'],
                            config['network_name']
                        )
                if state_dict:
                    network.load_state_dict(state_dict)
                return network
            return __make

        network_functions = {}
        for config in self.network_config:
            agent = config['agent']
            network_functions[agent] = _make(config)

        # cast to callable if return type is not dict
        if not self.return_dict:
            return network_functions['default']
        return network_functions
        
