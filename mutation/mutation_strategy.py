import numpy as np

from mutation.level_validator import LevelValidator


class MutationStrategy:
    def mutate(self, pair_list: list) -> list:
        raise NotImplementedError()


class EvolveStrategy(MutationStrategy):
    def __init__(
            self,
            level_validator: LevelValidator,
            max_children: int = 10,
            mutation_rate: float = 0.8,
            rand: np.random.RandomState = np.random.RandomState(),
    ):
        self._level_validator = level_validator
        self._max_children = max_children
        self._mutation_rate = mutation_rate
        self._rand = rand

    def mutate(self, pair_list: list) -> list:
        """Execute a mutation step of the existing generator_archive.

        The mutation strategy used here should be allowed to vary widely. For example, do we pick the environments
            which will be parents based on the ability of the agents in the current PAIR list? Do we randomly pick
            the parents? Do we pick parents based off of some completely other criteria that the user can set?

        :param pair_list: meta-population of Generators-Solvers (e.g. self.pairs in the POETManager class)
        :return:
        """
        child_list = []
        # we should be able to choose how the parents get selected. Increasing score? Decreasing score? User-defined?
        # set p in the np.random.choice function (leaving it blank is uniform probability).
        potential_parents = [pair_list[i] for i in self._rand.choice(len(pair_list), size=self._max_children)]

        for parent in potential_parents:
            new_generator = parent.generator.mutate(self._mutation_rate)
            if self._level_validator.validate_level(new_generator):
                child_list.append((parent.solver, new_generator))
                # todo track stats

        return child_list
