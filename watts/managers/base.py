import abc

from ..utils.register import Registrar
from ..gym_factory import GridGameFactory
from ..network_factory import NetworkFactory


class Manager(abc.ABC):
    def __init__(self, exp_name: str, gym_factory: GridGameFactory, network_factory: NetworkFactory, registrar: Registrar):
        """ABC for managers classes. Different managers will implement different algorithms
                e.g. POETManager\n
                e.g. PAIREDManager\n
                e.g. GPNManager\n

        :param exp_name: exp_name from launch script
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        :param registrar: class that dispenses necessary information e.g. num_poet_loops
        """

        self.registrar = registrar
        self.args = registrar.file_args
        self.exp_name = exp_name

        self.gym_factory = gym_factory
        self.network_factory = network_factory

    def evaluate(self) -> list:
        raise NotImplementedError

    def optimize(self) -> list:
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
