import numpy as np
from typing import List, Tuple
import unittest

from evolution.level_validator import AlwaysValidator
from evolution.evolution_strategy import BirthThenKillStrategy
from evolution.replacement_strategy import ReplaceOldest
from evolution.selection_strategy import SelectRandomly
from pair.agent_environment_pair import Pairing
from tests.test_classes import MockGenerator, MockPair, MockSolver


class TestEvolutionStrategy(unittest.TestCase):

    def test_single_selection_evolve(self):
        evolution_rate = 0.5
        evolve_strategy = BirthThenKillStrategy(
            level_validator=AlwaysValidator(),
            replacement_strategy=ReplaceOldest(),
            selection_strategy=SelectRandomly(max_children=1),
            evolution_rate=evolution_rate,
        )

        solver = MockSolver()
        generator = MockGenerator(evolution_rate)

        def birth_nop(children: List[Tuple]) -> List[Pairing]:
            return [Pairing(solver, generator) for _ in children]

        pairing = evolve_strategy.evolve(
            pair_list=[Pairing(solver, generator)],
            birth_func=birth_nop,
        )[0]

        assert solver == pairing.solver
        assert generator == pairing.generator

    def test_multi_selection_evolve(self):
        rand = np.random.RandomState(42)
        evolution_rate = 0.5
        evolve_strategy = BirthThenKillStrategy(
            level_validator=AlwaysValidator(),
            replacement_strategy=ReplaceOldest(),
            selection_strategy=SelectRandomly(max_children=5, rand=rand),
            evolution_rate=evolution_rate,
        )

        def birth_nop(children: List[Tuple]) -> List[Pairing]:
            return [Pairing(MockSolver(), MockGenerator(evolution_rate)) for _ in children]

        pairings = evolve_strategy.evolve(
            pair_list=[MockPair(MockSolver(), MockGenerator(evolution_rate)) for _ in range(3)],
            birth_func=birth_nop
        )

        assert len(pairings) == 5
