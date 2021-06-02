from generators.base import BaseGenerator
from solvers.base import BaseSolver


class Pairing:
    id = 0

    def __init__(self, solver: BaseSolver, generator: BaseGenerator):

        self.solver = solver
        self.generator = generator

        self.id = Pairing.id
        Pairing.id += 1

        self.results = []
        self.solved = False

    def __str__(self):
        return str(self.generator)

    def update_solver_weights(self, new_weights):
        self.solver.set_weights(new_weights)

    def get_solver_weights(self):
        return self.solver.get_weights()


if __name__ == "__main__":
    pass
