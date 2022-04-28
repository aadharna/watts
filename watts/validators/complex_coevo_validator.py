from typing import List, Tuple, Dict

from ..generators.base import BaseGenerator
from ..solvers.base import BaseSolver
from ..validators.level_validator import LevelValidator
from ..validators.agent_validator import ParentCutoffValidator, RandomAgentValidator
from ..validators.graph_validator import GraphValidator
from ..validators.Deepmind_validator import DeepMindAppendixValidator


class Foo(LevelValidator):
    def __init__(self, env_config, low_cutoff, high_cutoff, n_repeats, game_schema, network_factory_monad):
        """Instantiate a RandomAgentValidator, Deepmind_validator, and GraphValidator.
        Check for how well a random agent performs.
        Check for how well the trained agent performs relative to a baseline
        Check for if the level is even solvable by connectivity constraints

        @param env_config: config for loading levels into environemnt
        @param n_repeats: how many times should we evaluate the combo?

        @param env_config: env_config to save level string into
        @param low_cutoff: lower bound of what's too hard. default is -inf
        @param high_cutoff: upper bound of what is too easy. default is inf
        @param n_repeats: number of times to run an evaluate
        @param game_schema: What information do we need when checking connectivity of a game level?
        @param network_factory_monad: function that generates a rllib policy nn.
        """
        self.env_config = env_config
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.n_repeats = n_repeats
        self.game_schema = game_schema
        self.nfm = network_factory_monad

        self.random_agent_val = RandomAgentValidator(network_factory_monad=self.nfm,
                                                     env_config=self.env_config,
                                                     low_cutoff=self.low_cutoff,
                                                     high_cutoff=self.high_cutoff,
                                                     n_repeats=self.n_repeats)

        self.parent_cutoff_val = DeepMindAppendixValidator(env_config=self.env_config,
                                                           low_cutoff=self.low_cutoff,
                                                           n_repeats=self.n_repeats)

        self.graph_val = GraphValidator(self.game_schema)

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """

        :param generators: Generator class that we can extract a level string from
        :param solvers: Solver class that can play a game
        :param kwargs: future proofing
        @return:
        """
        random_solves, random_data = self.random_agent_val.validate_level(generators, solvers)
        parent_just_right, parent_data = self.parent_cutoff_val.validate_level(generators, solvers)
        is_solveable, graph_data = self.graph_val.validate_level(generators, solvers)

        data = {
            'ran_val': random_data,
            'pcv_val': parent_data,
            'graph_val': graph_data
        }

        if not random_solves and parent_just_right and is_solveable:
            return True, data
        else:
            return False, data
