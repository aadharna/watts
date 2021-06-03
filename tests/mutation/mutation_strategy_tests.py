import numpy as np
import unittest

from generators.base import BaseGenerator
from mutation.level_validator import AlwaysValidator
from mutation.mutation_strategy import EvolveStrategy
from pair.agent_environment_pair import Pair
from torch.nn import Module


class TestMutationStrategy(unittest.TestCase):
    class MockGenerator(BaseGenerator):
        def __init__(self, expected_mutation_rate: float):
            super().__init__()
            self._expected_mutation_rate = expected_mutation_rate

        def mutate(self, mutation_rate: float):
            assert self._expected_mutation_rate == mutation_rate
            return self

        def update_from_lvl_string(self, level_string):
            raise NotImplementedError

        def generate_fn_wrapper(self):
            def _generate() -> str:
                raise NotImplementedError
            return _generate

        def __str__(self):
            raise NotImplementedError

    class MockSolver(Module):
        pass

    def test_single_selection_evolve(self):
        mutation_rate = 0.5
        evolve_strategy = EvolveStrategy(AlwaysValidator(), max_children=1, mutation_rate=mutation_rate)

        solver = self.MockSolver()
        generator = self.MockGenerator(mutation_rate)

        result_solver, result_generator = evolve_strategy.mutate([Pair(solver, generator)])[0]

        assert solver == result_solver
        assert generator == result_generator

    def test_multi_selection_evolve(self):
        rand = np.random.RandomState(42)
        mutation_rate = 0.5
        evolve_strategy = EvolveStrategy(AlwaysValidator(), max_children=5, mutation_rate=mutation_rate, rand=rand)

        results = evolve_strategy.mutate([Pair(self.MockSolver(), self.MockGenerator(mutation_rate)) for _ in range(3)])

        solvers, generators = zip(*results)

        assert len(solvers) == 5
        assert len(generators) == 5
