from validators.level_validator import LevelValidator
from evolution.replacement_strategy import ReplaceOldest
from evolution.selection_strategy import SelectionStrategy
from typing import Callable, List, Tuple


class EvolutionStrategy:
    def evolve(self, active_population: list, birth_func) -> list:
        raise NotImplementedError()


class BirthThenKillStrategy(EvolutionStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            replacement_strategy: ReplaceOldest,
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
            if self._level_validator.validate_level(generator=new_generator, solver=parent.solver):
                children.append((parent.solver, new_generator, parent.id))

        children = birth_func(children)

        active_population.extend(children)

        return self._replacement_strategy.update(active_population)
