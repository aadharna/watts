import os
import networkx as nx
from networkx import grid_graph
import numpy as np

from generators.base import BaseGenerator


class LevelValidator:
    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        """Minimal Criteria for the newly created levels.
        In POET, this takes the form of agent ability on the newly created level
            Can the parent walker travel at least a minimum and not more than a maximum distance in the new map?
        In PINSKY, this takes the form of checking if a random agent can solve the level and if a "good" agent
            cannot solve the level (e.g. MCTS). In PINSKY, we used agents as the bounds to ensure the created
            level was playable.

        For example, we should be able to check the similarity of this level to existing levels and
            if they are "new" enough (e.g. novelty search), then it is an acceptable level.

        TODO: substantial validator (is this the right interface?)

        :param generator: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        raise NotImplementedError()


class AlwaysValidator(LevelValidator):
    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        """Passed generators are always valid.

        :param generator: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        return True


class RandomVariableValidator(LevelValidator):
    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        """Flips a coin on if we should use this generator or not.

        :param generator: generator object that contains a level.
        :return: boolean determining if the newly created level is allowed to exist
        """
        return np.random.rand() < 0.5


class RandomAgentValidator(LevelValidator):
    def __init__(self, gym_factory_monad, env_config):
        self.gf = gym_factory_monad
        self.config = env_config
        self.env = self.gf(self.config)

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        result = None
        _ = self.env.reset(level_string=str(generator))
        done = False
        while not done:
            ns, r, done, i = self.env.step(self.env.action_space.sample())
        if "PlayerResults" in i:
            result = i['PlayerResults']['1']
        return result == 'Win'


class GraphValidator(LevelValidator):
    """This will start with Zelda information coded in. Then we will generalize.

    """
    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        self.level_string = str(generator)
        level = self.level_string.split('\n')
        graph = grid_graph(dim=generator.shape)
        # remote blocking objects (e.g. walls, monsters, pits, etc)
        agent_start_pos = (0, 0)
        goal_pos = (0, 0)
        key_pos = []

        for j, row in enumerate(level):
            for i, char in enumerate(row):
                if char == "A":
                    agent_start_pos = (i, j)
                if char == 'g':
                    goal_pos = (i, j)
                if char == 'w' or char == 'e':
                    graph.remove_node((i, j))
                if char == '+':
                    key_pos.append((i, j))

        passable = False
        for key in key_pos:

            to_key = nx.has_path(graph,
                                 source=agent_start_pos,
                                 target=key)

            to_door = nx.has_path(graph,
                                  source=key,
                                  target=goal_pos)

            if to_key and to_door:
                passable = True
                self.start_to_key_length = nx.shortest_path_length(graph, agent_start_pos, key)
                self.key_to_goal_length = nx.shortest_path_length(graph, key, goal_pos)

        return passable


class PINSKYValidator(LevelValidator):
    def __init__(self, gym_factory_monad, env_config):
        self.random_agent_validator = RandomAgentValidator(gym_factory_monad, env_config)
        self.graph_validator = GraphValidator()

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        won_game_randomly = self.random_agent_validator.validate_level(generator)
        path_exists = self.graph_validator.validate_level(generator)

        if not won_game_randomly and path_exists:
            return True
        else:
            return False

"""
Example of a novelty validator useful for EC tests? (this can be removed until we support archives)

class NoveltyValidator(LevelValidator):
    def __init__(self, archive):
        self.archive = archive

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        is_novel = self.archive.is_new(generator)
        if is_novel:
            self.archive.add(generator)
        return is_novel
"""
