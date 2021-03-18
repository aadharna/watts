from managers.base import Manager

from evaluators.evaluate import evaluate_agent_on_level
from optimization.optimize import optimize_agent_on_env

from pair.agent_environment_pair import Pair
from generators.base import BaseGenerator

from gym_factory import GridGameFactory
from network_factory import NetworkFactory

class PoetManager(Manager):
    def __init__(self, exp_name: str, file_args, gym_factory: GridGameFactory, network_factory: NetworkFactory):
        """Extends the manager class to instantiate the POET algorithm

        :param exp_name: exp_name from launch script
        :param file_args: args loaded via utils.loader.load_from_yaml e.g. load_from_yaml(args.yaml)
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        """
        super().__init__(exp_name, file_args, gym_factory, network_factory)

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
                                               actor_critic_weights=p.solver.state_dict())
                for p in self.pairs]
        return refs

    def evaluate_combos(self) -> list:
        pass

    def mutate(self):
        pass

    def transfer(self):
        pass

    def optimize(self) -> list:
        """
        Optimize each NN-pair on its PAIRED environment
        :return: list of future-refs to the new optimized weights
        """
        refs = [optimize_agent_on_env.remote(trainer_constructor=self.gym_factory.registrar.trainer_constr,
                                             trainer_config=self.gym_factory.registrar.trainer_config,
                                             registered_gym_name=self.gym_factory.registrar.name,
                                             level_string=str(p.generator),
                                             actor_critic_weights=p.solver.state_dict())
                for p in self.pairs]
        return refs

    def pass_mc(self):
        pass

    def run(self):
        """
        # Paired Open Ended Trailblazer main loop
        #
        # For forever:
        #
        # 1) Build new environments (or change robot morphology)
        #    This could take the form of:
        #       * evolutionary algorithm
        #       * neural network
        #       * random domains
        #       * etc
        #
        #    1b) Make sure new environments are not too easy or too hard
        #    1c) When too many envs, remove envs that are too old (or some other metric)

        # 2) Optimize agents in their current environment
        #    This could be done with any sort of optimization algorithm
        #       * Evolutionary methods
        #       * RL
        #       * etc

        # 3) Transfer agents across environments
        #    This could be determined with
        #       * self-play evaluations
        #       * tournament selection
        #       * etc

        """
        pass