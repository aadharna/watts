import networkx as nx
from networkx import grid_graph

from generators.base import BaseGenerator
from solvers.base import BaseSolver
from validators.level_validator import LevelValidator


class GraphValidator(LevelValidator):
    """Does a path exist from the starting position to a key and then from the key to a door? Yes/No?
    Currently, this has game-specific information from Zelda and Maze.
    todo: generalize this.

    """
    def validate_level(self, generator: BaseGenerator, solver: BaseSolver, **kwargs) -> bool:
        """

        :param generator: Generator class that we can extract a level string from
        :param solver: n/a here; Solver class that can play a game
        :param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
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
                break

        return passable
