from evolution.replacement_strategy import ReplaceOldest
from tests.test_classes import MockGenerator, MockPair, MockSolver
import unittest


class TestReplacementStrategy(unittest.TestCase):

    def test_replace_oldest_cut_none(self):
        replacement_strategy = ReplaceOldest(max_pairings=4)
        archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(3)])
        assert(len(archive) == 3)

    def test_replace_oldest_cut_two(self):
        replacement_strategy = ReplaceOldest(max_pairings=4)
        archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(6)])
        alive_ids = [p.id for p in archive]
        assert(len(archive) == 4)
