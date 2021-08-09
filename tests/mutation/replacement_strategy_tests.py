from generators.base import BaseGenerator
from mutation.replacement_strategy import ReplaceOldest
from pair.agent_environment_pair import Pairing
from solvers.base import BaseSolver
import unittest


class MockPair(Pairing):
    def __init__(self, solver, generator):
        super().__init__(solver, generator)


class MockSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    @property
    def release(self):

        class Foo:
            def remote(self):
                return

        return Foo()


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


class TestReplacementStrategy(unittest.TestCase):

    def test_replace_oldest_cut_none(self):
        replacement_strategy = ReplaceOldest(max_pairings=4)
        archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(3)])
        assert(len(archive) == 3)

    def test_replace_oldest_cut_two(self):
        replacement_strategy = ReplaceOldest(max_pairings=4)
        archive = replacement_strategy.update([MockPair(MockSolver(), MockGenerator(0.01)) for i in range(6)])
        alive_ids = [p.id for p in archive]
        assert (len(archive) == 4)
        assert(alive_ids == [5, 4, 3, 2])

