from typing import Union

import ray

from watts.solvers.base import BaseSolver
from watts.generators.base import BaseGenerator
from watts.solvers.SingleAgentSolver import SingleAgentSolver


class Pairing:
    id = 0

    def __init__(self, solver: Union[BaseSolver, SingleAgentSolver], generator: BaseGenerator):
        """This class pairs together a solver and generator. This class also holds optimization and evaluation result
        information for the pair.

        Additionally, this can (but is not fully right now) act(ing)
        as a layer between the Manager and having to use ray directly.

        @param solver: watts.solvers class
        @param generator: watts.generator class
        """
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
        """
        Update the weights in this solver to these new weights
        @param new_weights: Dictionary of PyTorch tensor weights.
        @return:
        """
        self.solver.set_weights.remote(new_weights)

    def get_solver_weights(self):
        """
        get a reference to the weights from this solver
        @return:
        """
        return self.solver.get_weights.remote()

    def write_scaler_to_solver(self, name, value, loop):
        """write scaler data to graph

        @param name: what field is being appended to in the logger?
        @param value: What (real) value is being appeneded?
        @param loop: What is the x-axis of the graph? i.e., How many outer-loops are we on?
        @return:
        """
        self.solver.write.remote(name, value, loop)

    def get_eval_metric(self):
        """Get evaluation metric used in (traditional) evolution.
        @return:
        """
        # if eval_score list is populated, return the latest evaluation
        # Empty lists get evaluated as False. If that happens, return 0
        return self.eval_scores[-1] if self.eval_scores else 0

    def get_picklable_state(self):
        """
        Put the data into a picklable form.
        @return:
        """
        state_dict = {
            'solver': ray.get(self.solver.get_picklable_state.remote()),
            'generator': self.generator,
            'level_string': str(self.generator),
            'results': self.results,
            'solved': self.solved,
            'eval_scores': self.eval_scores,
            'pair_id': self.id
        }
        # the rllib Policy class is not picklable.
        #  If we remove it here, then we don't have to delete it from every single
        #  solver class. This is a result of using rllib::policy objects rather than raw pytorch::nns
        del state_dict['solver']['network_factory'].policy_class

        return state_dict
