import ray
from typing import Union

from generators.base import BaseGenerator
from solvers.base import BaseSolver
from solvers.SingleAgentSolver import SingleAgentSolver


class Pairing:
    id = 0

    def __init__(self, solver: Union[BaseSolver, SingleAgentSolver], generator: BaseGenerator):

        self.solver = solver
        self.generator = generator

        self.id = Pairing.id
        Pairing.id += 1

        self.results = []
        self.solved = []

    def __str__(self):
        return str(self.generator)

    def update_solver_weights(self, new_weights):
        self.solver.set_weights.remote(new_weights)

    def get_solver_weights(self):
        return self.solver.get_weights.remote()

    def serialize(self):
        return {
            'solver': ray.get(self.solver.get_picklable_state.remote()),
            'generator': self.generator,
            'level_string': str(self.generator),
            'results': self.results,
            'solved': self.solved,
            'pair_id': self.id
        }
