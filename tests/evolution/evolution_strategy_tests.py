import numpy as np
import unittest

from evolution.level_validator import AlwaysValidator
from evolution.evolution_strategy import BirthThenKillStrategy
from evolution.selection_strategy import SelectRandomly
from pair.agent_environment_pair import Pairing
from tests.test_classes import MockGenerator, MockPair, MockSolver


class TestEvolutionStrategy(unittest.TestCase):

    def test_single_selection_evolve(self):
        evolution_rate = 0.5
        evolve_strategy = BirthThenKillStrategy(
            AlwaysValidator(),
            SelectRandomly(max_children=1),
            evolution_rate=evolution_rate,
        )

        solver = MockSolver()
        generator = MockGenerator(evolution_rate)

        result_solver, result_generator, parent_id = evolve_strategy.evolve([Pairing(solver, generator)])[0]

        assert solver == result_solver
        assert generator == result_generator

    def test_multi_selection_evolve(self):
        rand = np.random.RandomState(42)
        evolution_rate = 0.5
        evolve_strategy = BirthThenKillStrategy(
            AlwaysValidator(),
            SelectRandomly(max_children=5, rand=rand),
            evolution_rate=evolution_rate,
        )

        results = evolve_strategy.evolve([MockPair(MockSolver(), MockGenerator(evolution_rate)) for _ in range(3)])

        solvers, generators, parent_ids = zip(*results)

        assert len(solvers) == 5
        assert len(generators) == 5
