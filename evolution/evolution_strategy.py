import numpy as np

from evolution.level_validator import LevelValidator


class EvolutionStrategy:
    def evolve(self, pair_list: list) -> list:
        raise NotImplementedError()


class BirthThenKillStrategy(EvolutionStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            max_children: int = 10,
            evolution_rate: float = 0.8,
            rand: np.random.RandomState = np.random.RandomState(),
    ):
        self._level_validator = level_validator
        self._max_children = max_children
        self._evolution_rate = evolution_rate
        self._rand = rand

    def evolve(self, pair_list: list) -> list:
        """Execute an evolution step of the existing generator_archive.

        The evolution strategy specifies how to combine Selection and Replacement.
        BirthThenKill first mutates parents to generate children using the SelectionStrategy,
        then kills them off using the ReplacementStrategy.

        :param pair_list: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :return:
        """
        child_list = []
        # we should be able to choose how the parents get selected. Increasing score? Decreasing score? User-defined?
        # set p in the np.random.choice function (leaving it blank is uniform probability).
        potential_parents = [pair_list[i] for i in self._rand.choice(len(pair_list), size=self._max_children)]

        for parent in potential_parents:
            new_generator = parent.generator.mutate(self._evolution_rate)
            if self._level_validator.validate_level(new_generator):
                child_list.append((parent.solver, new_generator, parent.id))

        return child_list
