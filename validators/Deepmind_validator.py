import numpy as np

from generators.base import BaseGenerator
from solvers.base import BaseSolver
from validators.agent_validator import RandomAgentValidator, ParentCutoffValidator
from validators.level_validator import LevelValidator


class DeepMindValidator(LevelValidator):
    """Is the new level too easy as measured by the parent's zero-shot performance on it
    and a random agent's performance? Yes/No?"""
    def __init__(self,
                 network_factory_monad,
                 env_config: dict,
                 low_cutoff: float = -np.inf,
                 high_cutoff: float = np.inf,
                 n_repeats: int = 1):
        """

        :param network_factory_monad: network factory build fn for the random agent validator
        :param env_config: env_config info for both validators
        :param low_cutoff: lower bound of what's too hard. default is -inf
        :param high_cutoff: upper bound of what is too easy. default is inf
        :param n_repeats: number of times to run an evaluate
        """
        self.random_agent_validator = RandomAgentValidator(network_factory_monad, env_config, n_repeats)
        self.parent_validator = ParentCutoffValidator(env_config, low_cutoff, high_cutoff, n_repeats)

    def validate_level(self, generator: BaseGenerator, solver: BaseSolver, **kwargs) -> bool:
        """

        :param generator: Generator class that we can extract a level string from
        :param solver: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        won_randomly = self.random_agent_validator.validate_level(generator=generator, solver=solver)
        not_too_easy = self.parent_validator.validate_level(generator=generator, solver=solver)
        if not won_randomly and not_too_easy:
            return True
        else:
            return False
