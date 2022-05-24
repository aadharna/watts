from typing import Dict, List, Tuple

import os
import ray

from watts.managers.base import Manager
from watts.utils.register import Registrar
from watts.utils.loader import save_obj
from watts.gym_factory import GridGameFactory
from watts.network_factory import NetworkFactory
from watts.transfer.rank_strategy import RankStrategy
from watts.pair.agent_environment_pair import Pairing
from watts.solvers.SingleAgentSolver import SingleAgentSolver
from watts.evolution.evolution_strategy import EvolutionStrategy
from watts.serializer.POETManagerSerializer import POETManagerSerializer


class PoetManager(Manager):
    """PoetManager runner.

    See:
     POET -- https://arxiv.org/abs/1901.01753
     Enhanced POET -- https://arxiv.org/abs/2003.08536
     PINSKY -- https://arxiv.org/abs/2007.08497
     PINSKY 2 -- https://arxiv.org/abs/2203.10941

    Each of the above algorithms implement a POET loop.
    """
    def __init__(
            self,
            exp_name: str,
            gym_factory: GridGameFactory,
            initial_pair: Pairing,
            evolution_strategy: EvolutionStrategy,
            transfer_strategy: RankStrategy,
            network_factory: NetworkFactory,
            registrar: Registrar,
            archive_dict: dict,
    ):
        """Extends the manager class to instantiate the POET algorithm

        @param exp_name: experiment name
        @param gym_factory: gym_factory class to create new learning envs
        @param initial_pair: pair::Pairing class to start the process
        @param evolution_strategy: strategy that dictates how the population of Pairs change over time
        @param transfer_strategy: strategy that dictates how the agents move between environments
        @param network_factory: network_factory class to create new NNs (Rllib Policies)
        @param registrar: a singleton class that holds information about the experiment
        @param archive_dict: serializable dictionary that contains no longer active agent-generator pairs
        """
        super().__init__(exp_name, gym_factory, network_factory, registrar)
        self._evolution_strategy = evolution_strategy
        self._transfer_strategy = transfer_strategy
        self.archive_dict = archive_dict
        self.active_population = [initial_pair]
        self.stats = {}
        self.stats['lineage'] = []
        self.stats['transfer'] = []
        self.i = 1

    def evaluate(self) -> list:
        """Evaluate each NN-pair on its PAIRED environment
        :return: list of future-refs to the evaluated objects
        """

        refs = []
        for p in self.active_population:
            config = self.registrar.get_config_to_build_rllib_env
            config['level_string'], _ = p.generator.generate_fn_wrapper()()
            refs.append(p.solver.evaluate.remote(env_config=config,
                                                 solver_id=p.id,
                                                 generator_id=p.id))

        return refs

    def optimize(self) -> list:
        """Optimize each NN-pair on its PAIRED environment
        :return: list of future-refs to the new optimized weights
        """

        refs = []
        for p in self.active_population:
            refs.append(p.solver.optimize.remote(trainer_config=self.registrar.get_trainer_config,
                                                 level_string_monad=p.generator.generate_fn_wrapper(),
                                                 pair_id=p.id))
        return refs

    def build_children(self, children: List[Tuple]) -> List[Pairing]:
        """Build a new Pairing class from the passed in list of agent-environment objects
        @param children: This is a list of (solver, generator, parent.id, novelty_ranking) to create new active agent-environment Pairings from
        @return: the list of initialized children
        """
        built_children = []
        for parent_solver, child_generator, parent_id, _ in children:
            parent_weights = ray.get(parent_solver.get_weights.remote())
            new_child = Pairing(solver=SingleAgentSolver.remote(trainer_constructor=self.registrar.trainer_constr,
                                                                trainer_config=self.registrar.trainer_config,
                                                                registered_gym_name=self.registrar.env_name,
                                                                network_factory=self.network_factory,
                                                                gym_factory=self.gym_factory,
                                                                weights=parent_weights,
                                                                log_id=f"{self.exp_name}_{Pairing.id}"),
                                generator=child_generator)
            built_children.append(new_child)
            self.stats['lineage'].append((parent_id, new_child.id))
        return built_children

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

        @return: void
        """
        while self.i <= self.args.num_poet_loops:
            i = self.i
            print(f"loop {i} / {self.args.num_poet_loops}")
            self.stats[i] = {}

            if i % self.args.evolution_timer == 0:
                self.active_population = self._evolution_strategy.evolve(
                    active_population=self.active_population,
                    birth_func=self.build_children
                )

            # this goes after the evolution_timer in case we add new objects to the active_population
            # helper dict to access all members of the active population by id
            # and then add things directly to their pair class state
            # this replaces all 4 of the previous `setter` helper functions previously
            active_populations = {p.id: p for p in self.active_population}

            opt_refs = self.optimize()
            opt_returns = ray.get(opt_refs)
            for opt_return in opt_returns:
                updated_weights = opt_return[0]['weights']
                pair_id = opt_return[0]['pair_id']
                return_dict = opt_return[0]['result_dict']
                active_populations[pair_id].update_solver_weights(updated_weights)
                active_populations[pair_id].results.append(return_dict)

            eval_refs = self.evaluate()
            eval_returns = ray.get(eval_refs)
            for eval_return in eval_returns:
                solved_status = eval_return[0]['win']
                eval_score = eval_return[0]['score']
                pair_id = eval_return['generator_id']
                active_populations[pair_id].solved.append(solved_status)
                active_populations[pair_id].eval_scores.append(eval_score)
                active_populations[pair_id].write_scaler_to_solver('paired_return', eval_score, self.i)

            if i % self.args.transfer_timer == 0:
                nets = [(p.solver, p.id) for j, p in enumerate(self.active_population)]
                lvls = [(p.generator, p.id) for j, p in enumerate(self.active_population)]
                new_weights = self._transfer_strategy.transfer(nets, lvls)

                for j, (best_w, best_id) in new_weights.items():
                    active_populations[j].update_solver_weights(best_w)
                    self.stats['transfer'].append((best_id, j, i))

            # Iterate i before snapshotting to avoid a duplicate loop iteration when loading from a snapshot
            self.i += 1

            if i % self.args.snapshot_timer == 0:
                POETManagerSerializer(self).serialize()
                # save raw data
                save_obj(self.archive_dict,
                         os.path.join('.', 'watts_logs', self.exp_name),
                         f'total_serialized_alg_{self.i}')
