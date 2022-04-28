from typing import List, Tuple, Dict

import numpy as np

from watts.solvers.base import BaseSolver
from watts.generators.base import BaseGenerator
from watts.validators.level_validator import LevelValidator
from watts.validators.agent_validator import RandomAgentValidator, ParentCutoffValidator, _eval_solver_on_generator


class DeepMindFullValidator(LevelValidator):
    """Is the new level too easy as measured by the parent's zero-shot performance on it
    and a random agent's performance? Yes/No?"""
    def __init__(self,
                 network_factory_monad,
                 env_config: dict,
                 low_cutoff: float = -np.inf,
                 high_cutoff: float = np.inf,
                 margin: float = 0.01,
                 n_tasks_difference_greater_than_margin: int = 2,
                 n_tasks_parent_greater_than_high: int = 4,
                 n_repeats: int = 5):
        """

        @param network_factory_monad: network factory build fn for the random agent validator
        @param env_config: env_config info for both validators
        @param low_cutoff: lower bound of what's too hard. default is -inf
        @param high_cutoff: upper bound of what is too easy. default is inf
        @param margin: By how much should the learned agent do better than the baseline? float = 0.01,
        @param n_tasks_difference_greater_than_margin: For how many games should the learned agent minus baseline do better than the margin? Must be less than n_repeats. int = 2,
        @param n_tasks_parent_greater_than_high: For how many games should the learned agent perform better the high_cutoff before the task is considered solved? int = 4,
        @param n_repeats: number of times to run an evaluate
        """
        self.n_repeats = n_repeats
        assert n_tasks_parent_greater_than_high <= n_repeats
        assert n_tasks_difference_greater_than_margin <= n_repeats
        # using deepmind's notation in the comments
        # this is m_>
        self.margin = margin
        # this is m_s
        self.high_cutoff = high_cutoff
        # this is m_cont
        self.low_cutoff = low_cutoff
        # this is m_>_cont
        self.p_just_right = n_tasks_difference_greater_than_margin / self.n_repeats
        # this is m_solved
        self.p_solved_cutoff = n_tasks_parent_greater_than_high / self.n_repeats

        self.random_agent_validator = RandomAgentValidator(network_factory_monad, env_config,
                                                           low_cutoff, high_cutoff, n_repeats)
        self.parent_validator = ParentCutoffValidator(env_config, low_cutoff, high_cutoff, n_repeats)

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """

        :param generators: Generator class that we can extract a level string from
        :param solvers: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        _, random_agent_data = self.random_agent_validator.validate_level(generators=generators, solvers=solvers)
        _, parent_data = self.parent_validator.validate_level(generators=generators, solvers=solvers)

        # on average, the random policy scores less than low_cutoff
        #   a control policy return threshold 𝑚cont = 5 would [remove] any training tasks where a control policy is
        #   able to get a return of at least 5
        too_easy_for_random = np.any(random_agent_data['scores'] >= self.low_cutoff)
        # on average, does the parent do better than random by some margin
        #   The combination, for example, of
        #   𝑚> = 2 and 𝑚>cont = 0.9 would only allow training tasks
        #   where the agent achieves a return in all ten episode samples
        #   of at least 2 reward more than the return achieved by the
        #   control policy
        #
        # we don't collapse the runs yet. We want to know on how many runs did we exceed
        # the margin by? This should be vector subtraction.
        difference = parent_data['scores'] - random_agent_data['scores']
        parent_is_better_than_random = np.mean(difference >= self.margin) > self.p_just_right
        # agent has low probability of scoring high on a given task
        #   As a final example, the combination of
        #   𝑚s = 450 and 𝑚solved = 0.1 would disallow training on
        #   any task where the agent is able to achieve more than 450
        #   reward on any of its episode samples
        not_too_easy_for_parent = np.mean(parent_data['scores'] >= self.high_cutoff) < self.p_solved_cutoff

        data = {
            'ran_val': random_agent_data,
            'pcv_val': parent_data,
        }

        if not too_easy_for_random and parent_is_better_than_random and not_too_easy_for_parent:
            return True, data
        else:
            return False, data


class DeepMindAppendixValidator(LevelValidator):
    """Is the new level too easy or too hard as measured by the parent's zero-shot performance on it?
     Yes/No?"""
    def __init__(self,
                 env_config: dict,
                 low_cutoff: float = -np.inf,
                 n_repeats: int = 100):
        """

        :param env_config: env_config info for both validators
        :param low_cutoff: lower bound of what's too hard. default is -inf
        :param n_repeats: number of times to run an evaluate
        """
        self.config = env_config
        self.low_cutoff = low_cutoff
        self.n_repeats = n_repeats

    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """

        :param generators: Generator class that we can extract a level string from
        :param solvers: Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        scores = []
        wins = []
        for n in range(self.n_repeats):
            win, score, _ = _eval_solver_on_generator(generator=generators[0], solver=solvers[0], config=self.config)
            scores.append(score)
            wins.append(win)
        # if agent scores equal to or better than low_cutoff at least once
        scores = np.array(scores)
        not_too_hard = np.any(scores >= self.low_cutoff)
        # agent scores equal to or better than low_cutoff on less than or equal to half of its rollouts
        not_too_easy = np.sum(scores >= self.low_cutoff) <= (self.n_repeats // 2)
        return not_too_easy and not_too_hard, {'scores': np.array(scores),
                                               'wins': np.array(wins)}
