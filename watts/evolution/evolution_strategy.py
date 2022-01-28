from typing import Callable, List, Tuple

from ..validators.level_validator import LevelValidator
from .selection_strategy import SelectionStrategy
from .replacement_strategy import ReplaceOldest, ReplacementStrategy, _release
from ..transfer.rank_strategy import RankStrategy
from ..validators.rank_novelty_validator import RankNoveltyValidator


class EvolutionStrategy:
    def evolve(self, active_population: list, birth_func) -> list:
        raise NotImplementedError()


class BirthThenKillStrategy(EvolutionStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            replacement_strategy: ReplacementStrategy,
            selection_strategy: SelectionStrategy,
            mutation_rate: float = 0.8,
    ):
        self._level_validator = level_validator
        self._replacement_strategy = replacement_strategy
        self._selection_strategy = selection_strategy
        self._mutation_rate = mutation_rate

    def evolve(self, active_population: list, birth_func: Callable[[List[Tuple]], List]) -> list:
        """Execute an evolution step of the existing generator_archive.

        The evolution strategy specifies how to combine Selection and Replacement.
        BirthThenKill first mutates parents to generate children using the SelectionStrategy,
        then kills them off using the ReplacementStrategy.

        :param active_population: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :param birth_func: a function describing how new pairs are created
        :return:
        """
        children = []
        potential_parents = self._selection_strategy.select(active_population)

        for parent in potential_parents:
            new_generator = parent.generator.mutate(self._mutation_rate)
            is_valid, data = self._level_validator.validate_level(generators=[new_generator], solvers=[parent.solver])
            if is_valid:
                children.append((parent.solver, new_generator, parent.id, 0))

        children = birth_func(children)

        active_population.extend(children)

        return self._replacement_strategy.update(active_population)


class TraditionalES(EvolutionStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            replacement_strategy: ReplacementStrategy,
            selection_strategy: SelectionStrategy,
            mutation_rate: float = 0.8,
    ):
        self._level_validator = level_validator
        self._replacement_strategy = replacement_strategy
        self._selection_strategy = selection_strategy
        self._mutation_rate = mutation_rate

    def evolve(self, active_population: list, birth_func: Callable[[List[Tuple]], List]) -> list:
        """Execute an evolution step of the existing generator_archive.

        The evolution strategy specifies how to combine Selection and Replacement.
        TraditionalES (without replacement) selects parents to be part of the next generation and does not replace them
        Then we "fill out" the rest of the population (n - k) individuals

        :param active_population: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :param birth_func: a function describing how new pairs are created
        :return:
        """
        children = []
        proto_children = []
        # remove (n-k) least performant individuals
        parents = self._replacement_strategy.update(active_population)
        # keep k elites
        children.extend(parents)
        # spawn (n-k) individuals
        potential_parents = self._selection_strategy.select(parents)
        for parent in potential_parents:
            new_generator = parent.generator.mutate(self._mutation_rate)
            proto_children.append((parent.solver, new_generator, parent.id, 0))
        children.extend(birth_func(proto_children))
        return children


class POETStrategy(EvolutionStrategy):
    """This is the method by which Wang et al adjust their active population in POET.
    https://arxiv.org/abs/1901.01753 see page 27.
    For their code, see: https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/poet_algo.py#L282
    This has been adapted to watts from the uber-ai code so it will not look 1-1, but the behaviours are replicated.

    This implementation has also been generalized so it doesn't have walker specific parts anymore.
    """
    def __init__(
            self,
            level_validator: LevelValidator,
            replacement_strategy: ReplacementStrategy,
            selection_strategy: SelectionStrategy,
            transfer_strategy: RankStrategy,
            env_config: dict,
            network_factory,
            env_factory,
            historical_archive: dict,
            density_threshold: float = 1.0,
            k: int = 3,
            low_cutoff: float = 0.,
            high_cutoff: float = 150.,
            mutation_rate: float = 0.8,
    ):
        self._level_validator = level_validator
        self._replacement_strategy = replacement_strategy
        self._selection_strategy = selection_strategy
        self._mutation_rate = mutation_rate
        self._transfer_strategy = transfer_strategy
        self._novelty_validator = RankNoveltyValidator(density_threshold=density_threshold,
                                                       env_config=env_config,
                                                       historical_archive=historical_archive,
                                                       agent_factory=network_factory,
                                                       env_factory=env_factory,
                                                       k=k,
                                                       low_cutoff=low_cutoff,
                                                       high_cutoff=high_cutoff)
        self.data = {}

    def evolve(self, active_population: list, birth_func) -> list:

        nets = [(p.solver, p.id) for j, p in enumerate(active_population)]
        solvers, solver_idxs = zip(*nets)
        generators = [p.generator for p in active_population]

        # Hacky...
        # update the embeddings in the novelty validator
        # todo unhack this
        for g in generators:
            self._novelty_validator.calculate_pata_ec(generator=g, solvers=solvers)

        for archived_pair_id, archived_pair in self._novelty_validator.historical_archive.items():
            self._novelty_validator.calculate_pata_ec(generator=archived_pair['generator'], solvers=solvers)


        # this will already limit us to max potential children
        potential_parents = self._selection_strategy.select(active_population)
        # mutate the parents to get potential children
        proto_children = self._get_child_list(potential_parents, active_population=active_population)
        # create the remote objects. It's annoying to create these objects in full here
        #  since we're going to cull some of them away in the next few lines, but oh well.
        children = birth_func(proto_children)
        # create container for transfer learning step; we'll reuse this for each potential child
        to_remove = []
        for i, child in enumerate(children):
            # find best weights for child via a transfer learning step
            #  when compared against the current population
            new_weights = self._transfer_strategy.transfer(solver_list=nets,
                                                           generator_list=[(child.generator, child.id)])
            for g_id, (best_w, s_id) in new_weights.items():
                child.update_solver_weights(best_w)
            # check if the proposed pair is worth learning on
            is_valid, data = self._level_validator.validate_level(generators=[child.generator], solvers=[child.solver])
            if not is_valid:
                # release the resources that child i claimed and save pointer to this child so we can stop it from
                # joining the active population
                _release({}, [child])
                to_remove.append(i)

        alive_children = []
        for i, child in enumerate(children):
            if i not in to_remove:
                alive_children.append(child)
        active_population.extend(alive_children)
        return self._replacement_strategy.update(active_population)

    def _get_child_list(self, parent_list, active_population):
        child_list = []
        active_solvers = [p.solver for p in active_population]
        active_generators = [p.generator for p in active_population]
        for parent in parent_list:
            new_generator = parent.generator.mutate(mutation_rate=self._mutation_rate)
            is_valid, data = self._level_validator.validate_level(generators=[new_generator], solvers=[parent.solver])
            if is_valid:
                is_novel, novelty_data = self._novelty_validator.validate_level(solvers=active_solvers,
                                                                                generators=active_generators,
                                                                                proposed_generator=new_generator)
                child_list.append((parent.solver, new_generator, parent.id, novelty_data.get('top_k_mean', 0)))

        # sort child list according to novelty from high to low
        child_list = sorted(child_list, key=lambda x: x[3], reverse=True)
        return child_list
