import abc

from gym_factory import GridGameFactory
from network_factory import NetworkFactory


class Manager(abc.ABC):
    def __init__(self, exp_name: str, file_args, gym_factory: GridGameFactory, network_factory: NetworkFactory):
        """ABC for managers classes. Different managers will implement different algorithms
                e.g. POETManager
                e.g. PAIREDManager
                e.g. GPNManager

        :param exp_name: exp_name from launch script
        :param file_args: args loaded via utils.loader.load_from_yaml e.g. load_from_yaml(args.yaml)
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        """
        self.args = file_args
        self.exp_name = exp_name

        self.gym_factory = gym_factory
        self.network_factory = network_factory

    def evaluate(self) -> list:
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
