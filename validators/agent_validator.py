import copy
import numpy as np
import ray
from typing import Tuple

from generators.base import BaseGenerator
from solvers.base import BaseSolver
from validators.level_validator import LevelValidator


def _eval_solver_on_generator(generator: BaseGenerator, solver: BaseSolver, config: dict) -> Tuple[bool, float]:
    """

    :param generator: Generator class object that whose string representation is the level in question
    :param solver: (remote) Solver class object that we can issue an evaluate call on
    :param config: game config so that we can update the level in the remote class
    :return: (win, score) boolean float tuple
    """
    config = copy.deepcopy(config)
    config['level_string'] = str(generator)
    result = ray.get(solver.evaluate.remote(config, solver_id=0, generator_id=0))
    key = ray.get(solver.get_key.remote())
    score = result[key]['score']
    win = result[key]['win']
    return win, score


def _opt_solver_on_generator(generator: BaseGenerator, solver: BaseSolver, config: dict) -> dict:
    """Run one step of optimization of this solver on this generator.

    This would be useful if you wanted to do something like:
    # >>> solver_weights = solver.get_weights()
    # >>> zero_shot_score = evaluate(generator, solver)
    # >>> opt_result = _opt_solver_on_generator(generator, solver, config)
    # >>> one_shot_score = evaluate(generator, solver)
    # >>> solver.set_weights(solver_weights)
    # >>> score_difference = one_shot_score - zero_shot_score
    # >>> if score_difference >= 0 return True else False

    :param generator: Generator class that we can extract a level string from
    :param solver: Solver class that can play a game
    :param config: config for us to load the level into to broadcast to the remote workers
    :return: True/False is this level a good level to use?
    """
    config = copy.deepcopy(config)
    config['level_string'] = str(generator)
    opt_result = ray.get(solver.optimize.remote(config, generator.generate_fn_wrapper()))
    return opt_result


class RandomAgentValidator(LevelValidator):
    """Can a random agent win this level with multiple tries? Yes/No?
    """
    def __init__(self, network_factory_monad, env_config, n_repeats=1):
        """

        :param network_factory_monad: function to build NNs with
        :param env_config: config to load a level into
        :param n_repeats: number of times to run the evaluate
        """
        self.nf = network_factory_monad
        self.config = env_config
        self.n_repeats = n_repeats

    def validate_level(self, generator: BaseGenerator, solver: BaseSolver, **kwargs) -> bool:
        """Load random weights into the solver and run an evaluate. Then restore the solver to its original state.

        :param generator: Generator class that we can extract a level string from
        :param solver: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        wins = []
        scores = []
        # extract the solver's weights
        solver_weights = ray.get(solver.get_weights.remote())
        for _ in range(self.n_repeats):
            # get random agent
            random_agent = self.nf({})
            random_weights = random_agent.state_dict()
            solver.set_weights.remote(random_weights)
            win, score = _eval_solver_on_generator(generator, solver, self.config)
            wins.append(win)
            scores.append(score)
        # set the solvers weights back to what it was before the random agent
        solver.set_weights.remote(solver_weights)
        return any(wins)


class ParentCutoffValidator(LevelValidator):
    """On average, is the new level too easy as measured by the parent's zero-shot performance on it? Yes/No?

    This is the POETValidator.
    """
    def __init__(self, env_config: dict, low_cutoff: float = -np.inf, high_cutoff: float = np.inf, n_repeats=1):
        """

        :param env_config: env_config to save level string into
        :param low_cutoff: lower bound of what's too hard. default is -inf
        :param high_cutoff: upper bound of what is too easy. default is inf
        :param n_repeats: number of times to run an evaluate
        """
        self.config = env_config
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.n_repeats = n_repeats

    def validate_level(self, generator: BaseGenerator, solver: BaseSolver, **kwargs) -> bool:
        """Run the passed in solver on the passed in generator.

        :param generator: Generator class that we can extract a level string from
        :param solver: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        scores = []
        wins = []
        for n in range(self.n_repeats):
            win, score = _eval_solver_on_generator(generator=generator, solver=solver, config=self.config)
            scores.append(score)
            wins.append(win)
        return self.low_cutoff <= np.mean(scores) <= self.high_cutoff
