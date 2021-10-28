from generators.base import BaseGenerator
from solvers.base import BaseSolver
from validators.agent_validator import RandomAgentValidator
from validators.graph_validator import GraphValidator
from validators.level_validator import LevelValidator

from typing import List


class PINSKYValidator(LevelValidator):
    """Minimal Criterion used in the "Co-generating game levels and game playing agents" paper.
    https://arxiv.org/abs/2007.08497
    Almost; Instead of a graph validation, we used an MCTS agent, but that's not supported at
    the moment, so we're doing a search on a graph now as a comparable thing.

    """
    def __init__(self, network_factory_monad, env_config, n_repeats):
        """

        :param network_factory_monad: network factory build fn for the random agent validator
        :param env_config: env_config info for random agent validator
        :param n_repeats: number of times to run an evaluate on random agent validator
        """
        self.random_agent_validator = RandomAgentValidator(network_factory_monad, env_config, n_repeats)
        self.graph_validator = GraphValidator()

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> bool:
        """

        :param generators: Generator class that we can extract a level string from
        :param solvers: n/a here; Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        won_game_randomly = self.random_agent_validator.validate_level(generators, solvers)
        path_exists = self.graph_validator.validate_level(generators, solvers)

        if not won_game_randomly and path_exists:
            return True
        else:
            return False
