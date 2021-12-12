from ray.rllib.models import ModelCatalog
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent

from .models.AIIDE_network import AIIDEActor
from .models.PCGRL_network import PCGRLAdversarial

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
    def __init__(self, network_name: str, nn_build_info: dict):
        """Factory to create NNs and register it with ray's global NN register

        :param registrar: utils.registery.Registrar object. This holds various dicts needed for initialization.

        """
        self.nn_build_info = nn_build_info
        self.network_name = network_name
        self.constructor = get_network_constructor(self.network_name)

        ModelCatalog.register_custom_model(self.network_name, self.constructor)

    def make(self):
        """Make a pytorch NN.

        :param state_dict: Dictionary containing the state to initialize this N with. If empty, uses default state.
        :return: a pytorch network
        """
        def _make(state_dict):
            network = self.constructor(**self.nn_build_info)
            if state_dict:
                network.load_state_dict(state_dict)
            return network
        return _make
