import copy
from typing import Tuple, List, Dict

import ray
import numpy as np
import torch

from ..solvers.base import BaseSolver
from ..generators.base import BaseGenerator
from ..validators.level_validator import LevelValidator
from ..utils.returns import compute_gae


def _eval_solver_on_generator(generator: BaseGenerator, solver: BaseSolver, config: dict) -> Tuple[bool, float, dict]:
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
    return win, score, result[key]['kwargs']


def _get_solver_value_of_generator(generator: BaseGenerator, solver: BaseSolver) -> float:
    return ray.get(solver.value_function.remote(level_string=str(generator)))


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
    def __init__(self, network_factory_monad, env_config, low_cutoff, high_cutoff, n_repeats=1):
        """

        :param network_factory_monad: function to build NNs with
        :param env_config: config to load a level into
        :param n_repeats: number of times to run the evaluate
        """
        self.nf = network_factory_monad
        self.config = env_config
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.n_repeats = n_repeats

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """Load random weights into the solver and run an evaluate. Then restore the solver to its original state.

        :param generators: Generator class that we can extract a level string from
        :param solvers: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        wins = []
        scores = []
        # extract the solver's weights
        solver_weights = ray.get(solvers[0].get_weights.remote())
        for _ in range(self.n_repeats):
            # get random agent
            random_agent = self.nf({})
            random_weights = random_agent.state_dict()
            solvers[0].set_weights.remote(random_weights)
            win, score, _ = _eval_solver_on_generator(generators[0], solvers[0], self.config)
            wins.append(win)
            scores.append(score)
        # set the solvers weights back to what it was before the random agent
        solvers[0].set_weights.remote(solver_weights)
        return self.low_cutoff <= np.mean(scores) <= self.high_cutoff, {'wins': np.array(wins),
                                                                        'scores': np.array(scores)}


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

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """Run the passed in solver on the passed in generator.

        :param generators: Generator class that we can extract a level string from
        :param solvers: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        assert len(generators) == 1
        scores = []
        wins = []
        for n in range(self.n_repeats):
            win, score, _ = _eval_solver_on_generator(generator=generators[0], solver=solvers[0], config=self.config)
            scores.append(score)
            wins.append(win)
        return self.low_cutoff <= np.mean(scores) <= self.high_cutoff, {'wins': np.array(wins),
                                                                        'scores': np.array(scores)}


class ParentValueValidator(LevelValidator):
    """Do we expect this new level to get us positive return? Yes/No?
    """

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        value = _get_solver_value_of_generator(generators[0], solvers[0])
        return 0 <= value, {'value': value}


class PositiveGAEValidator(LevelValidator):
    """On average, for this agent, does this level induce positive GAE returns? We use this to approximate
    regret as based on: https://openreview.net/pdf?id=rRg0ghtqRw2
    which is in turn based on: https://arxiv.org/abs/2010.03934

    """
    def __init__(self, env_config, n_repeats: int = 5, gamma: float = 0.99, tau: float = 0.95):
        self.env_config = env_config
        self.n_repeats = n_repeats
        self.gamma = gamma
        self.tau = tau

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        gaes = []
        for i in range(self.n_repeats):
            _, _, return_kwargs = _eval_solver_on_generator(generators[0], solvers[0], self.env_config)
            returns = compute_gae(next_value=0,
                                  rewards=return_kwargs['rewards'],
                                  masks=return_kwargs['dones'],
                                  values=return_kwargs['values'],
                                  gamma=self.gamma,
                                  tau=self.tau)
            returns = torch.cat(returns).double()
            average_gae_loss = torch.mean(torch.where(returns > 0., returns, 0.)).item()
            gaes.append(average_gae_loss)

        return np.mean(gaes) >= 0, {'gae_loss': gaes}


class PositiveRegretMultiAgentValidator(LevelValidator):
    """On average, does this level induce positive regret between a pair of agents?
    Let the first agent be the "positive" agent (i.e. a1_score - a2_score >= 0)
    """
    def __init__(self, env_config, n_repeats: int = 3):
        self.n_repeats = n_repeats
        self.env_config = env_config

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        a1_score, a1_wins = [], []
        a2_score, a2_wins = [], []
        for n in range(self.n_repeats):
            win1, score1, _ = _eval_solver_on_generator(generators[0], solvers[0], config=self.env_config)
            win2, score2, _ = _eval_solver_on_generator(generators[0], solvers[1], config=self.env_config)
            a1_score.append(score1)
            a2_score.append(score2)
        return np.mean(a1_score) - np.mean(a2_score) >= 0, {'a1_scores': np.array(a1_score),
                                                            'a2_scores': np.array(a2_score),
                                                            'a1_wins': np.array(a1_wins),
                                                            'a2_wins': np.array(a2_wins)}
