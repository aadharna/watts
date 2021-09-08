from evolution.level_validator import LevelValidator
from evolution.selection_strategy import SelectionStrategy


class EvolutionStrategy:
    def evolve(self, pair_list: list) -> list:
        raise NotImplementedError()


class BirthThenKillStrategy(EvolutionStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            selection_strategy: SelectionStrategy,
            evolution_rate: float = 0.8,
    ):
        self._level_validator = level_validator
        self._selection_strategy = selection_strategy
        self._evolution_rate = evolution_rate

    def evolve(self, pair_list: list) -> list:
        """Execute an evolution step of the existing generator_archive.

        The evolution strategy specifies how to combine Selection and Replacement.
        BirthThenKill first mutates parents to generate children using the SelectionStrategy,
        then kills them off using the ReplacementStrategy.

        :param pair_list: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :return:
        """
        child_list = []
        potential_parents = self._selection_strategy.select(pair_list)

        for parent in potential_parents:
            new_generator = parent.generator.mutate(self._evolution_rate)
            if self._level_validator.validate_level(new_generator):
                child_list.append((parent.solver, new_generator, parent.id))

        return child_list
