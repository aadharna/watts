import ray
from typing import Dict, Any

from gym_factory import GridGameFactory
from managers.base import Manager
from mutation.mutation_strategy import MutationStrategy
from mutation.replacement_strategy import ReplacementStrategy
from network_factory import NetworkFactory
from pair.agent_environment_pair import Pairing
from serializer.POETManagerSerializer import POETManagerSerializer
from solvers.SingleAgentSolver import SingleAgentSolver
from transfer.rank_strategy import RankStrategy
from utils.register import Registrar


class PoetManager(Manager):
    def __init__(
            self,
            exp_name: str,
            gym_factory: GridGameFactory,
            initial_pair: Pairing,
            mutation_strategy: MutationStrategy,
            replacement_strategy: ReplacementStrategy,
            transfer_strategy: RankStrategy,
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
        self._transfer_strategy = transfer_strategy
        self._replacement_strategy = replacement_strategy
        self.active_population = [initial_pair]
        self.archive_history = self._replacement_strategy.archive_history
        self.stats = {}
        self.stats['lineage'] = []
        self.stats['transfer'] = []
        self.i = 1

    def set_solver_weights(self, pair_id: int, new_weights: Dict):
        for p in self.active_population:
            if p.id == pair_id:
                p.update_solver_weights(new_weights)

    def append_solver_result(self, pair_id: int, result_dict: Dict):
        for p in self.active_population:
            if p.id == pair_id:
                p.results.append(result_dict)

    def set_win_status(self, pair_id: int, generator_solved_status: bool):
        for p in self.active_population:
            if p.id == pair_id:
                p.solved = generator_solved_status
                break

    def evaluate(self) -> list:
        """
        Evaluate each NN-pair on its PAIRED environment
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
        """
        Optimize each NN-pair on its PAIRED environment
        :return: list of future-refs to the new optimized weights
        """

        refs = []
        for p in self.active_population:
            refs.append(p.solver.optimize.remote(trainer_config=self.registrar.get_trainer_config,
                                                 level_string_monad=p.generator.generate_fn_wrapper(),
                                                 pair_id=p.id))
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
        while self.i <= self.args.num_poet_loops:
            i = self.i
            print(f"loop {i} / {self.args.num_poet_loops}")
            self.stats[i] = {}

            if i % self.args.mutation_timer == 0:
                children = self._mutation_strategy.mutate(self.active_population)
                for solver, generator, parent_id in children:
                    weights = ray.get(solver.get_weights.remote())
                    self.active_population.append(Pairing(solver=SingleAgentSolver.remote(trainer_constructor=self.registrar.trainer_constr,
                                                                                          trainer_config=self.registrar.trainer_config,
                                                                                          registered_gym_name=self.registrar.env_name,
                                                                                          network_factory=self.network_factory,
                                                                                          gym_factory=self.gym_factory,
                                                                                          weights=weights),
                                                          generator=generator))
                    self.stats['lineage'].append((parent_id, self.active_population[-1].id))

                self.active_population = self._replacement_strategy.update(self.active_population)

            opt_refs = self.optimize()
            opt_returns = ray.get(opt_refs)
            for opt_return in opt_returns:
                updated_weights = opt_return[0]['weights']
                pair_id = opt_return[0]['pair_id']
                self.set_solver_weights(pair_id, updated_weights)
                self.append_solver_result(pair_id, opt_return[0]['result_dict'])

            eval_refs = self.evaluate()
            eval_returns = ray.get(eval_refs)
            for eval_return in eval_returns:
                solved_status = eval_return[0]['win']
                pair_id = eval_return['generator_id']
                self.set_win_status(pair_id, solved_status)

            if i % self.args.transfer_timer == 0:
                nets = [(p.solver, j) for j, p in enumerate(self.active_population)]
                lvls = [(p.generator, j) for j,  p in enumerate(self.active_population)]
                id_map = [p.id for p in self.active_population]
                new_weights = self._transfer_strategy.transfer(nets, lvls, id_map=id_map)

                for j, (best_w, best_id) in new_weights.items():
                    self.set_solver_weights(j, best_w)
                    self.stats['transfer'].append((best_id, j, i))

            # Iterate i before snapshotting to avoid a duplicate loop iteration when loading from a snapshot
            self.i += 1

            if i % self.args.snapshot_timer == 0:
                POETManagerSerializer(self).serialize()
