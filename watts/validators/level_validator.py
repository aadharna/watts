from typing import List, Tuple, Dict

import numpy as np

from ..solvers.base import BaseSolver
from ..generators.base import BaseGenerator


class LevelValidator:
    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """Minimal Criteria for the newly created levels.
        In POET, this takes the form of agent ability on the newly created level
            Can the parent walker travel at least a minimum and not more than a maximum distance in the new map?
        In PINSKY, this takes the form of checking if a random agent can solve the level and if a "good" agent
            cannot solve the level (e.g. MCTS). In PINSKY, we used agents as the bounds to ensure the created
            level was playable.

        For example, we should be able to check the similarity of this level to existing levels and
            if they are "new" enough (e.g. novelty search), then it is an acceptable level.

        TODO: substantial validator (is this the right interface?)

        :param solvers:
        :param generators: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        raise NotImplementedError()


class AlwaysValidator(LevelValidator):
    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """Passed generators are always valid.

        :param solvers:
        :param generators: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        return True, {}


class RandomVariableValidator(LevelValidator):
    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """Flips a coin on if we should use this generator or not.

        :param solvers:
        :param generators: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        return np.random.rand() < 0.5, {}


"""
Example of a novelty validator useful for EC tests? (this can be removed until we support archives)

class NoveltyValidator(LevelValidator):
    def __init__(self, archive):
        self.archive = archive

    def validate_level(self, generator: BaseGenerator, solver: BaseSolver, **kwargs) -> bool:
        is_novel = self.archive.is_new(generator)
        if is_novel:
            self.archive.add(generator)
        return is_novel
"""
