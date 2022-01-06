import pytest

from watts.evolution.selection_strategy import SelectRandomly
from watts.pair.agent_environment_pair import Pairing

from tests.test_classes import MockGenerator, MockPair, MockSolver



def test_single_random_selection():
    mutation_rate = 0.5
    generator = MockGenerator(mutation_rate)
    solver = MockSolver()

    selection_strategy = SelectRandomly(max_children=1)

    pairing = selection_strategy.select([Pairing(solver, generator)])[0]

    assert solver == pairing.solver
    assert generator == pairing.generator

def test_multi_random_selection():
    mutation_rate = 0.5
    selection_strategy = SelectRandomly(max_children=3)

    pairings = selection_strategy.select([MockPair(MockSolver(), MockGenerator(mutation_rate)) for _ in range(5)])

    assert len(pairings) == 3
