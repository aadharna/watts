from typing import Callable, List, Tuple

from ..validators.level_validator import LevelValidator
from .selection_strategy import SelectionStrategy
from .replacement_strategy import ReplaceOldest, ReplacementStrategy


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
            if self._level_validator.validate_level(generators=[new_generator], solvers=[parent.solver]):
                children.append((parent.solver, new_generator, parent.id))

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
            proto_children.append((parent.solver, new_generator, parent.id))
        children.extend(birth_func(proto_children))
        return children