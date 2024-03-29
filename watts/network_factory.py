from typing import Callable

from ray.rllib.models import ModelCatalog
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent

from watts.models.AIIDE_network import AIIDEActor
from watts.models.PCGRL_network import PCGRLAdversarial
from watts.models.FC_network import TwoLayerFC

aiide = "AIIDE_PINSKY_MODEL"
conv = "SimpleConvAgent"
gap = "GAPAgent"
pcgrl = "Adversarial_PCGRL"
fc = 'TwoLayerFC'


def get_network_constructor(network_name: str):
    """
    Get network constructor by name
    @param network_name: network type name
    @return:
    """
    if network_name == aiide:
        return AIIDEActor
    elif network_name == pcgrl:
        return PCGRLAdversarial
    elif network_name == conv:
        return SimpleConvAgent
    elif network_name == gap:
        return GAPAgent
    elif network_name == fc:
        return TwoLayerFC
    else:
        raise ValueError("Network unavailable. Add the network definition to the models folder and network_factory")


class NetworkFactory:
    def __init__(self, network_name: str, nn_build_info: dict, policy_class: Callable):
        """
        Factory to create NNs and register it with ray's global NN register

        @param network_name: name of the network desired.
        @param nn_build_info: necessary build info (ObsSpace, ActSpace, etc)
        @param policy_class: Rllib policy class constructor/caller
        """
        self.nn_build_info = nn_build_info
        self.network_name = network_name
        self.constructor = get_network_constructor(self.network_name)
        self.policy_class = policy_class

        ModelCatalog.register_custom_model(self.network_name, self.constructor)

    def make(self):
        """Make an RLlib Policy Class that wraps a Pytorch NN.

        :param state_dict: Dictionary containing the state to initialize this NN with. If empty, uses default state.
        :return: a pytorch network
        """
        def _make(state_dict):
            policy = self.policy_class(**self.nn_build_info)
            if state_dict:
                policy.model.load_state_dict(state_dict)
            return policy
        return _make
