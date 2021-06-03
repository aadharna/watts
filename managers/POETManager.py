import numpy as np
import ray

from evaluators.remote_evaluate import async_evaluate_agent_on_level
from gym_factory import GridGameFactory
from itertools import product
from managers.base import Manager
from mutation.mutation_strategy import MutationStrategy
from network_factory import NetworkFactory
from optimization.optimize import optimize_agent_on_env
from pair.agent_environment_pair import Pair
from typing import Dict, Any
from utils.register import Registrar


class PoetManager(Manager):
    def __init__(
            self,
            exp_name: str,
            gym_factory: GridGameFactory,
            initial_pair: Pair,
            mutation_strategy: MutationStrategy,
            network_factory: NetworkFactory,
            registrar: Registrar,
    ):
        """Extends the manager class to instantiate the POET algorithm

        :param exp_name: exp_name from launch script
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        :param registrar: class that dispenses necessary information e.g. num_poet_loops
        """
        super().__init__(exp_name, gym_factory, network_factory, registrar)
        self._mutation_strategy = mutation_strategy
        self.pairs = [initial_pair]
        self.forgotten_pairs = []

    def evaluate(self) -> list:
        """
        Evaluate each NN-pair on its PAIRED environment
        :return: list of future-refs to the evaluated objects
        """
        refs = [async_evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                                     rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                                     level_string_monad=p.generator.generate_fn_wrapper(),
                                                     network_factory_monad=self.network_factory.make(),
                                                     actor_critic_weights=p.solver.state_dict(),
                                                     solver_id=p.id,
                                                     generator_id=p.id)
                for p in self.pairs]
        return refs

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
                solver_id, generator_id = id_map[i]
                ref = async_evaluate_agent_on_level.remote(gym_factory_monad=self.gym_factory.make(),
                                                           rllib_env_config=self.registrar.get_config_to_build_rllib_env,
                                                           level_string_monad=g,
                                                           network_factory_monad=self.network_factory.make(),
                                                           actor_critic_weights=s,
                                                           solver_id=solver_id,
                                                           generator_id=generator_id)
                refs.append(ref)
            return refs

        solvers, solver_idxs = zip(*solver_list)
        generators, generator_idxs = zip(*generator_list)
        solver_generator_combo_id = list(product(id_map, id_map))
        combo_refs = evaluate_combos(solvers=[s.state_dict() for s in solvers],
                                     generators=[g.generate_fn_wrapper() for g in generators],
                                     id_map=solver_generator_combo_id)

        # n_gens = len(generators)
        # n_solvs = len(solvers)
        results = ray.get(combo_refs)
        # returned_scores = []
        # returned_generators = []
        # returned_solvers = []
        new_weights = {}
        for i, generator_id in enumerate(id_map):
            best_s = -np.inf
            best_w = solvers[i].state_dict()
            best_id = generator_id
            for j, r in enumerate(results):

                if generator_id == r['generator_id']:
                    score = sum(r['score'])
                    solver = r['solver_id']
                    if score > best_s:
                        best_s = score
                        best_w = solvers[id_map.index(solver)].state_dict()
                        best_id = solver
            new_weights[generator_id] = (best_w, best_id)
            # todo track this info for analysis purposes
            # print(f"updated {generator_id} to {best_id}")



        #         returned_scores.append(sum(r['score']))
        #         returned_generators.append(r['generator_id'])
        #         returned_solvers.append(r['solver_id'])
        #
        # scores = np.array(returned_scores).reshape(n_gens, n_solvs).T  # this will be square for POET
        # scores[2][2] = 100
        # gens = np.array(returned_generators).reshape(n_gens, n_solvs)
        # solvs = np.array(returned_solvers).reshape(n_gens, n_solvs)
        # best_result_index_per_gen = np.argmax(scores, axis=1)
        #
        #
        # for i, (best_pair_id, generator_id) in enumerate(zip(best_result_index_per_gen, id_map)):
        #     # todo change these indicies into the pair.id to track who is leaving where and going to where
        #
        #     # if id_map @ best_solver_index == generator_id
        #     #   then we would just be replacing ourself, so there is no need to track a transfer
        #     best_solver = id_map.index(solvs[i][best_pair_id])
        #     new_weights[generator_id] = solvers[best_solver].state_dict()

        return new_weights

    def optimize(self) -> list:
        """
        Optimize each NN-pair on its PAIRED environment
        :return: list of future-refs to the new optimized weights
        """
        refs = [optimize_agent_on_env.remote(trainer_constructor=self.registrar.trainer_constr,
                                             trainer_config=self.registrar.trainer_config,
                                             registered_gym_name=self.registrar.name,
                                             level_string_monad=p.generator.generate_fn_wrapper(),
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
                children = self._mutation_strategy.mutate(self.pairs)
                for solver, generator in children:
                    self.pairs.append(Pair(solver, generator))

                if len(self.pairs) > self.args.max_envs:
                    aged_pairs = sorted(self.pairs, key=lambda x: x.id, reverse=True)
                    self.pairs = aged_pairs[:self.args.max_envs]
                    self.forgotten_pairs.extend(aged_pairs[self.args.max_envs:])
                    del aged_pairs

            opt_refs = self.optimize()
            opt_returns = ray.get(opt_refs)
            for opt_return in opt_returns:
                updated_weights = opt_return['weights']
                pair_id = opt_return['pair_id']
                self.set_solver_weights(pair_id, updated_weights)
                self.pairs[pair_id].results.append(opt_return['result_dict']) # TODO does this work as expected?

            eval_refs = self.evaluate()
            eval_returns = ray.get(eval_refs)
            for eval_return in eval_returns:
                solved_status = eval_return['win']
                pair_id = eval_return['generator_id']
                self.set_win_status(pair_id, solved_status)

            if i % self.args.transfer_timer:
                nets = [(p.solver, i) for i, p in enumerate(self.pairs)]
                lvls = [(p.generator, i) for i,  p in enumerate(self.pairs)]
                id_map = [p.id for p in self.pairs]
                new_weights = self.transfer(nets, lvls, id_map=id_map)

                for j, (best_w, best_id) in new_weights.items():
                    self.set_solver_weights(j, best_w)


if __name__ == "__main__":
    pass
