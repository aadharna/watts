import copy
import ray
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
    """Can a random agent win this level? Yes/No?"""
    def __init__(self, gym_factory_monad, env_config):
        self.gf = gym_factory_monad
        self.config = env_config
        self.env = self.gf(self.config)

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        result = None
        _ = self.env.reset(level_string=str(generator))
        done = False
        self.reward = 0
        while not done:
            ns, r, done, i = self.env.step(self.env.action_space.sample())
            self.reward += r
        if "PlayerResults" in i:
            result = i['PlayerResults']['1']
        return result == 'Win'


class RepeatedRandomAgentValidator(RandomAgentValidator):
    """Can a random agent win this level with multiple tries? Yes/No?
    """
    def __init__(self, gym_factory_monad, env_config, n_repeats):
        RandomAgentValidator.__init__(self, gym_factory_monad, env_config)
        self.n_repeats = n_repeats

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        scores = []
        wins = []
        for n in range(self.n_repeats):
            wins.append(super().validate_level(generator=generator))
            scores.append(self.reward)
        return any(wins)  # too easy if a random agent wins


class GraphValidator(LevelValidator):
    """Does a path exist from the starting position to a key and then from the key to a door? Yes/No?
    Currently, this has game-specific information from Zelda and Maze.
    todo: generalize this.

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
                if char == 'g' or char == 'x':
                    goal_pos = (i, j)
                if char == 'w' or char == 'e' or char == 't':
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
    """Minimal Criterion used in the Co-generating game levels and game playing agents paper.
    Almost; Instead of a graph validation, we used an MCTS agent, but that's not supported at
    the moment, so we're doing a search on a graph now as a comparable thing.

    """
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


class ParentCutoffValidator(LevelValidator):
    """Is the new level too easy as measured by the parent's zero-shot performance on it? Yes/No?"""
    def __init__(self, env_config, cutoff):
        self.config = env_config
        self.cutoff = cutoff

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        solver = kwargs.get('solver', None)
        config = copy.deepcopy(self.config)
        config['level_string'] = str(generator)
        self.result = ray.get(solver.evaluate.remote(config, solver_id=0, generator_id=0))
        key = ray.get(solver.get_key.remote())
        self.score = self.result[key]['score']
        return self.score <= self.cutoff  # ensure the new level is not too easy for the parent


class RepeatedParentCutoffValidator(ParentCutoffValidator):
    """On average, is the new level too easy as measured by the parent's zero-shot performance on it? Yes/No?"""
    def __init__(self, env_config, cutoff, n_repeats=5):
        ParentCutoffValidator.__init__(self, env_config, cutoff)
        self.n_repeats = n_repeats

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        solver = kwargs.get('solver', None)
        if solver is None:
            raise ValueError('need to pass (parent) solver to Parent Cutoff Validator')
        scores = []
        for n in range(self.n_repeats):
            yes = super().validate_level(generator=generator, solver=solver)
            scores.append(self.score)
        return np.mean(scores) <= self.cutoff


class POETValidator(ParentCutoffValidator):
    """Is the new level too easy or too hard as measured by the parent's zero-shot performance on it? Yes/No?"""
    def __init__(self, env_config, high_cutoff, low_cutoff):
        ParentCutoffValidator.__init__(self, env_config, cutoff=0)
        # we will functionally override this singular cutoff in the base ParentCutoffValidator manually here
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        solver = kwargs.get('solver', None)
        if solver is None:
            raise ValueError('need to pass (parent) solver to POET Validator')
        _ = super().validate_level(generator=generator, solver=solver)
        # check we can do better than a minimal cutoff value
        # check we cannot do better than a maximal cutoff value
        # If both true, the level is good enough
        if self.low_cutoff <= self.score <= self.high_cutoff:
            return True
        else:
            return False


class DeepMindValidator(LevelValidator):
    """Is the new level too easy as measured by the parent's zero-shot performance on it
    and a random agent's performance? Yes/No?"""
    def __init__(self, gym_factory_monad, env_config, cutoff, n_repeats):
        self.random_agent_validator = RandomAgentValidator(gym_factory_monad, env_config)
        self.parent_validator = RepeatedParentCutoffValidator(env_config, cutoff, n_repeats)

    def validate_level(self, generator: BaseGenerator, **kwargs) -> bool:
        solver = kwargs.get('solver', None)
        if solver is None:
            raise ValueError('need to pass (parent) solver to DeepMind Validator')
        won_randomly = self.random_agent_validator.validate_level(generator)
        not_too_easy = self.parent_validator.validate_level(generator=generator,
                                                            solver=solver)
        if not won_randomly and not_too_easy:
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
