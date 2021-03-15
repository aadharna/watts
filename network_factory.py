import gym

from models.AIIDE_network import AIIDEActor
from models.PCGRL_networks import PCGRLAdversarial
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent


class NetworkFactory:
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space,
                         num_outputs: int, model_config: dict = {}, name: str = 'AIIDE_PINSKY_MODEL'):
        """Factory to create NNs

        :param name: Name of NN you want this factory to create. MUST be a valid NN. See models.AIIDE_network for eg
        :param obs_space: gym.space for observations
        :param action_space: gym.space for actions
        :param num_outputs: number of output actions
        :param model_config: model_config for rllib
        """
        self.obs = obs_space
        self.acs = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name

        self.constructor = self.get_network_constructor()

    def get_network_constructor(self):
        if self.name == "AIIDE_PINSKY_MODEL":
            return AIIDEActor
        elif self.name == "Adversarial_PCGRL":
            return PCGRLAdversarial
        elif self.name == "SimpleConvAgent":
            return SimpleConvAgent
        elif self.name == "GAPAgent":
            return GAPAgent
        else:
            raise ValueError("Network unavailable. Add the network definition to the models folder and network_factory")

    def make(self):
        def _make():
            return self.constructor(self.obs, self.acs, self.num_outputs, self.model_config, self.name)
        return _make

if __name__ == "__main__":
    import os
    from utils.loader import load_from_yaml
    from utils.register import register_env_with_rllib
    args = load_from_yaml('args.yaml')

    name, nActions, actSpace, obsSpace, observer = register_env_with_rllib(file_args=args)

    network_factory = NetworkFactory(obs_space=obsSpace,
                                     action_space=actSpace,
                                     num_outputs=nActions,
                                     model_config={},
                                     name=args.network_name)

    network = network_factory.make()()
    print(network)