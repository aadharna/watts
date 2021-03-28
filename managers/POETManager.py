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
        self.archive = []

    def add_pair(self, network, generator: BaseGenerator):
        self.pairs.append(Pair(network, generator))

    def evaluate(self) -> list:
        """
        Evaluate each NN-pair on its PAIRED environment
        :return: list of future-refs to the evaluated objects
        """
        refs = [evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                               rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                               level_string=p.generator.generate(),
                                               network_factory_monad=self.network_factory.make(),
                                               actor_critic_weights=p.solver.state_dict())
                for p in self.pairs]
        return refs

    def mutate(self, pair_list: list) -> list:
        """Execute a mutation step of the existing generator_archive.

        The mutation strategy used here should be allowed to vary widely. For example, do we pick the environments
            which will be parents based on the ability of the agents in the current PAIR list? Do we randomly pick
            the parents? Do we pick parents based off of some completely other criteria that the user can set?

        :param pair_list: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :return:
        """
        def pass_mc(generator) -> bool:
            """Minimal Criteria for the newly created levels.
            In POET, this takes the form of agent ability on the newly created level
                Can the parent walker travel at least a minimum and not more than a maximum distance in the new map?
            In PINSKY, this takes the form of checking if a random agent can solve the level and if a "good" agent
                cannot solve the level (e.g. MCTS). In PINSKY, we used agents as the bounds to ensure the created
                level was playable.

            The MC should be its own whole modular piece that is part of the evolutionary half of Engima.

            For example, we should be able to check the similarity of this level to existing levels and
                if they are "new" enough (e.g. novelty search), then it is an acceptable level.

            #todo MODULARLIZE THIS INTO ITS OWN PIECE and add functionality!

            :param generator: generator object that contains a level.
            :return: boolean determining if the newly created level is allowed to exist
            """
            return True if np.random.rand() < 0.5 else False

        child_list = []
        # we should be able to choose how the parents get selected. Increasing score? Decreasing score? User-defined?
        # set p in the np.random.choice function (leaving it blank is uniform probability).
        potential_parents = [pair_list[i] for i in np.random.choice(len(pair_list),
                                                                    size=self.args.max_children)]

        for parent in potential_parents:
            new_generator = parent.generator.mutate(self.args.mutation_rate)
            if pass_mc(new_generator):
                child_list.append((parent.solver, new_generator))
                # todo track stats

        return child_list

    def set_solver_weights(self, pair_id: int, new_weights: Dict):
        for p in self.pairs:
            if p.id == pair_id:
                p.update_solver_weights(new_weights)

    def set_win_status(self, pair_id: int, generator_solved_status: bool):
        for p in self.pairs:
            if p.id == pair_id:
                p.solved = generator_solved_status
                break

    def transfer(self, solver_list: list, generator_list: list, id_map: list) -> Dict[int, Any]:
        """Run the transfer tournament; take in a solver list and a generator list.

        todo modularize this

        This should be its own modular piece. For example, in this instantiation I am trying all combinations
            of agents and environments to find the best agents for each generator via a tournament update.

        However, it should be relatively simple to instead:
            - use self-play scores
            or
            - implement this paper: https://arxiv.org/pdf/1806.02643.pdf as the transfer mechanism.
            The above paper notes that the fundamental algebraic structure of tournaments and evaluation is
            antisymmetric.

        :return: dict of new weights indexed by a pair_id.
        """
        def evaluate_combos(solvers, generators, id_map) -> list:
            """run the evaluate function on the cartesian product of solvers and generators

            :param solvers: list of NN-weights
            #TODO: make this a callable that will generate the string via e.g. NN-weights for a generator network
            :param generators: list of callable string levels
            :param id_map: list of tuples of (solver_id, generator_id) also created with product (id, id)
            :return: list of refs (and list of ids) for the evaluated solver-generator pairs
            """

            refs = []
            for i, (s, g) in enumerate(product(solvers, generators)):

                # todo: I don't like this. We need a better way of getting the solver/generator ids.
                # this is the same to do as below.
                solver_id = id_map[i][0]
                generator_id = id_map[i][1]
                refs.append(evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                                           rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                                           level_string=g,
                                                           network_factory_monad=self.network_factory.make(),
                                                           actor_critic_weights=s,
                                                           solver_id=solver_id,
                                                           generator_id=generator_id))
            return refs

        combo_refs = evaluate_combos(solvers=[s.state_dict() for s in solver_list],
                                     generators=[g.generate() for g in generator_list],
                                     id_map=id_map)

        results = ray.get(combo_refs)
        scores = [sum(r['score']) for i, r in enumerate(results)]
        scores = np.array(scores).reshape(len(solver_list), len(generator_list))  # this will be square for POET
        best_ids = np.argmax(scores, axis=1)
        new_weights = {}
        for i, best_pair_id in enumerate(best_ids):
            # todo change these indicies into the pair.id to track who is leaving where and going to where
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
                                             level_string_monad=p.generator.generate(),
                                             actor_critic_weights=p.solver.state_dict(),
                                             pair_id=p.id)
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
        for i in range(self.args.num_poet_loops):
            print(f"loop {i} / {self.args.num_poet_loops}")

            if i % self.args.mutation_timer:
                children = self.mutate(self.pairs)
                for solver, generator in children:
                    self.add_pair(solver, generator)

                if len(self.pairs) > self.args.max_envs:
                    aged_pairs = sorted(self.pairs, key=lambda x: x.id, reverse=True)
                    self.pairs = aged_pairs[:self.args.max_envs]
                    self.archive.extend(aged_pairs[self.args.max_envs:])
                    del aged_pairs

            opt_refs = self.optimize()
            opt_returns = ray.get(opt_refs)
            for opt_return in opt_returns:
                updated_weights = opt_return['weights']
                pair_id = opt_return['pair_id']
                self.set_solver_weights(pair_id, updated_weights)
                self.pairs[pair_id].returns.append(opt_return['result_dict'])

            eval_refs = self.evaluate()
            eval_returns = ray.get(eval_refs)
            for eval_return in eval_returns:
                solved_status = eval_return['win']
                pair_id = eval_return['gen_id']
                self.set_win_status(pair_id, solved_status)

            if i % self.args.transfer_timer:
                nets = [p.solver for p in self.pairs]
                lvls = [p.generator for p in self.pairs]
                # todo fix this
                id_map = list(product([p.id for p in self.pairs], [p.id for p in self.pairs]))
                new_weights = self.transfer(nets, lvls, id_map=id_map)

                for j, new_weight in new_weights.items():
                    self.set_solver_weights(j, new_weight)


if __name__ == "__main__":
    pass
