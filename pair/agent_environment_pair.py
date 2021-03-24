from generators.base import BaseGenerator
from torch.nn import Module

class Pair:
    id = 0

    def __init__(self, agent: Module, generator: BaseGenerator):

        self.solver = agent
        self.generator = generator

        self.id = Pair.id
        Pair.id += 1

        self.results = []
        self.solved = False

    def __str__(self):
        return str(self.generator)

    def update_solver_weights(self, new_weights):
        self.solver.load_state_dict(new_weights)


if __name__ == "__main__":
    pass
