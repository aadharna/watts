import numpy as np
import unittest

from mutation.level_validator import AlwaysValidator
from mutation.mutation_strategy import EvolveStrategy
from pair.agent_environment_pair import Pairing
from tests.test_classes import MockGenerator, MockPair, MockSolver


class TestMutationStrategy(unittest.TestCase):

    def test_single_selection_evolve(self):
        mutation_rate = 0.5
        evolve_strategy = EvolveStrategy(AlwaysValidator(), max_children=1, mutation_rate=mutation_rate)

        solver = MockSolver()
        generator = MockGenerator(mutation_rate)

        result_solver, result_generator, parent_id = evolve_strategy.mutate([Pairing(solver, generator)])[0]

        assert solver == result_solver
        assert generator == result_generator

    def test_multi_selection_evolve(self):
        rand = np.random.RandomState(42)
        mutation_rate = 0.5
        evolve_strategy = EvolveStrategy(AlwaysValidator(), max_children=5, mutation_rate=mutation_rate, rand=rand)

        results = evolve_strategy.mutate([MockPair(MockSolver(), MockGenerator(mutation_rate)) for _ in range(3)])

        solvers, generators, parent_ids = zip(*results)

        assert len(solvers) == 5
        assert len(generators) == 5
