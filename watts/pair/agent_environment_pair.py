from typing import Union

import ray

from ..solvers.base import BaseSolver
from ..generators.base import BaseGenerator
from ..solvers.SingleAgentSolver import SingleAgentSolver


class Pairing:
    id = 0

    def __init__(self, solver: Union[BaseSolver, SingleAgentSolver], generator: BaseGenerator):

        self.solver = solver
        self.generator = generator

        self.id = Pairing.id
        Pairing.id += 1

        self.results = []
        self.solved = []
        self.eval_scores = []

    def __str__(self):
        return str(self.generator)

    def update_solver_weights(self, new_weights):
        self.solver.set_weights.remote(new_weights)

    def get_solver_weights(self):
        return self.solver.get_weights.remote()

    def get_eval_metric(self):
        # if eval_score list is populated, return the latest evaluation
        # Empty lists get evaluated as False. If that happens, return 0
        return self.eval_scores[-1] if self.eval_scores else 0

    def get_picklable_state(self):
        return {
            'solver': ray.get(self.solver.get_picklable_state.remote()),
            'generator': self.generator,
            'level_string': str(self.generator),
            'results': self.results,
            'solved': self.solved,
            'eval_scores': self.eval_scores,
            'pair_id': self.id
        }
