from ray.rllib.models import ModelCatalog

from models.AIIDE_network import AIIDEActor
from models.PCGRL_networks import PCGRLAdversarial
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent


class NetworkFactory:
    def __init__(self, registrar):
        """Factory to create NNs and register it with ray's global NN register

        :param registrar: utils.registery.Registrar object. This holds various dicts needed for initialization.

        """
        self.registrar = registrar
        self.network_name = self.registrar.network_name
        self.constructor = self.get_network_constructor()

        ModelCatalog.register_custom_model(self.network_name, self.constructor)

    def get_network_constructor(self):
        if self.network_name == "AIIDE_PINSKY_MODEL":
            return AIIDEActor
        elif self.network_name == "Adversarial_PCGRL":
            return PCGRLAdversarial
        elif self.network_name == "SimpleConvAgent":
            return SimpleConvAgent
        elif self.network_name == "GAPAgent":
            return GAPAgent
        else:
            raise ValueError("Network unavailable. Add the network definition to the models folder and network_factory")

    def make(self):
        """Make a pytorch NN.

        :return: function to build a pytorch network
        """
        def _make():
            return self.constructor(**self.registrar.get_nn_build_info)
        return _make


if __name__ == "__main__":
    from utils.loader import load_from_yaml
    from utils.register import Registrar
    args = load_from_yaml('args.yaml')

    registry = Registrar(file_args=args)

    network_factory = NetworkFactory(registrar=registry)

    network = network_factory.make()()
    print(network)
