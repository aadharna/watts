import ray
import numpy as np
from typing import Tuple, Dict, Any

from managers.base import Manager

from evaluators.evaluate import evaluate_agent_on_level
from optimization.optimize import optimize_agent_on_env

from pair.agent_environment_pair import Pair
from generators.base import BaseGenerator

from gym_factory import GridGameFactory
from network_factory import NetworkFactory
from utils.register import Registrar

from itertools import product


class PoetManager(Manager):
    def __init__(self, exp_name: str, gym_factory: GridGameFactory, network_factory: NetworkFactory, registrar: Registrar):
        """Extends the manager class to instantiate the POET algorithm

        :param exp_name: exp_name from launch script
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        :param registrar: class that dispenses necessary information e.g. num_poet_loops
        """
        super().__init__(exp_name, gym_factory, network_factory, registrar)
        self.pairs = []

    def add_pair(self, network, generator: BaseGenerator):
        self.pairs.append(Pair(self.args, network, generator))

    def evaluate(self) -> list:
        """
        Evaluate each NN-pair on its PAIRED environment
        :return: list of future-refs to the evaluated objects
        """
        refs = [evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                               rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                               level_string=str(p.generator),
                                               network_factory_monad=self.network_factory.make(),
                                               actor_critic_weights=p.solver.state_dict())
                for p in self.pairs]
        return refs

    def mutate(self):
        pass

    def pass_mc(self):
        pass

    def set_solver_weights(self, pair_id: int, new_weights: Dict):
        self.pairs[pair_id].update_solver_weights(new_weights)

    def transfer(self, solver_list: list, generator_list: list) -> Dict[int, Any]:
        """Run the transfer tournament; take in a solver list and a generator list.

        :return: dict of new weights indexed by a pair_id.
        """
        def evaluate_combos(solvers, generators) -> list:
            """run the evaluate function on the cartesian product of solvers and generators

            :param solvers: list of NN-weights
            #TODO: make this a callable that will generate the string via e.g. NN-weights for a generator network
            :param generators: list of string levels
            :return: list of refs (and list of ids) for the evaluated solver-generator pairs
            """

            refs = []
            for i, (s, g) in enumerate(product(solvers, generators)):
                refs.append(evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                                           rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                                           level_string=g,
                                                           network_factory_monad=self.network_factory.make(),
                                                           actor_critic_weights=s))
                # ref_ids.append((i // (len(solvers)), i % (len(generators))))
            return refs  # , ref_ids

        combo_refs = evaluate_combos(solvers=[s.state_dict() for s in solver_list],
                                     generators=[str(g) for g in generator_list])

        results = ray.get(combo_refs)
        scores = [sum(r['score']) for i, r in enumerate(results)]
        scores = np.array(scores).reshape(len(solver_list), len(generator_list))  # this will be square for POET
        best_ids = np.argmax(scores, axis=1)
        new_weights = {}
        for i, best_pair_id in enumerate(best_ids):
            new_weights[i] = solver_list[best_pair_id].state_dict()

        return new_weights

    def optimize(self) -> list:
        """
        Optimize each NN-pair on its PAIRED environment
        :return: list of future-refs to the new optimized weights
        """
        refs = [optimize_agent_on_env.remote(trainer_constructor=self.registrar.trainer_constr,
                                             trainer_config=self.registrar.trainer_config,
                                             registered_gym_name=self.registrar.name,
                                             level_string=str(p.generator),
                                             actor_critic_weights=p.solver.state_dict())
                for p in self.pairs]
        return refs

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


if __name__ == "__main__":
    pass
