from utils.loader import load_from_yaml
from utils.evaluate import evaluate_agent_on_level

from pair.agent_environment_pair import Pair
from generators.base import BaseGenerator

from gym_factory import GridGameFactory
from network_factory import NetworkFactory

class PoetManager:
    def __init__(self, exp_name: str, file_args, gym_factory: GridGameFactory, network_factory: NetworkFactory):
        """

        :param exp_name: exp_name from launch script
        :param file_args: args loaded via utils.loader.load_from_yaml e.g. load_from_yaml(args.yaml)
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        """
        self.args = file_args
        self.exp_name = exp_name

        self.gym_factory = gym_factory
        self.network_factory = network_factory

        self.pairs = []

    def add_pair(self, network, generator: BaseGenerator):
        self.pairs.append(Pair(self.args, network, generator))

    def evaluate(self) -> list:
        """
        Evaluate each NN-pair on its PAIRED environment
        :return: list of future-refs to the evaluated objects
        """
        refs = [evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                               network_factory_monad=self.network_factory.make(),
                                               level_string=str(p.generator),
                                               actor_critic_weights=p.agent.state_dict())
                for p in self.pairs]
        return refs

    def evaluate_combos(self) -> list:
        pass

    def mutate(self):
        pass

    def transfer(self):
        pass

    def optimize(self):
        pass

    def pass_mc(self):
        pass