import pytest
from watts.evolution.replacement_strategy import ReplaceOldest, KeepTopK

from tests.test_classes import MockGenerator, MockPair, MockSolver


def test_replace_oldest_cut_none():
    replacement_strategy = ReplaceOldest(max_pairings=4)
    archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(3)])
    assert(len(archive) == 3)

def test_replace_oldest_cut_two():
    replacement_strategy = ReplaceOldest(max_pairings=4)
    archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(6)])
    alive_ids = [p.id for p in archive]
    assert(len(archive) == 4)

def test_keep_single_best():
    mutation_rate = 0.5
    generator = MockGenerator(mutation_rate)
    solver = MockSolver()

    selection_strategy = KeepTopK(max_pairings=1)

    pairing = selection_strategy.update([MockPair(solver, generator)])[0]

    assert solver == pairing.solver
    assert generator == pairing.generator

def test_keep_multi_top3():
    mutation_rate = 0.5
    selection_strategy = KeepTopK(max_pairings=3)
    pairings = [MockPair(MockSolver(), MockGenerator(mutation_rate)) for _ in range(5)]
    for p, score in zip(pairings, [2, 4, 3, 5, 1]):
        p.eval_scores.append(score)
    selected = selection_strategy.update(pairings)

    assert len(selected) == 3
    assert selected[0].get_eval_metric() == 5
    assert selected[1].get_eval_metric() == 4
    assert selected[2].get_eval_metric() == 3

def test_keep_more_than_built():
    mutation_rate = 0.5
    selection_strategy = KeepTopK(max_pairings=10)
    pairings = [MockPair(MockSolver(), MockGenerator(mutation_rate)) for _ in range(5)]
    selected = selection_strategy.update(pairings)
    assert len(selected) == 5
    assert selected == pairings
