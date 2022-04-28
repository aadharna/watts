from typing import List, Tuple, Dict

import networkx as nx
from networkx import grid_graph

from watts.solvers.base import BaseSolver
from watts.game.GameSchema import GameSchema
from watts.generators.base import BaseGenerator
from watts.validators.level_validator import LevelValidator


class GraphValidator(LevelValidator):

    def __init__(self, game_schema: GameSchema):
        self.game_schema = game_schema

    """Graph Search a level to determine if there is a way to win it.

    In order to make this validation generic across games, we make a few assumptions:
    1. The game has 1 or more agents.
    2. The agents must reach some object in order to "win".

    Different games may have multiple objects that must be reached in order to win.
    We're not certain beforehand which objects these are, so we assume that all non-wall objects
    must be reachable by all agents in order for the level to be valid - we consider these "interesting".

    This restriction disqualifies some levels that are winnable, for example if a monster
    is unable to be reached like in this limited zelda level:
    w w w w w
    w A + g w
    w w w w w
    w . e . w
    w w w w w
    
    This also disqualifies levels only winnable by some mechanics, for example in limited
    zelda you only need to reach one key, so levels like this are winnable:
    w w w w w
    w A w . w
    w + w + w
    w g w . w
    w w w w w
    However, an equivalent level of endless_butterflies wouldn't be winnable since you need
    to capture ALL the butterflies:
    w w w w w
    w A w . w
    w b w b w
    w b w . w
    w w w w w
    We opt for the more generic mechanics described above to handle these discrepancies at
    the cost of failing more levels than may be strictly necessary.
    
    Of course, you can define your own graph validator based off this that perhaps encodes
    more game-specific information if that'd be useful for you.
    """
    def validate_level(self,  generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """

        @param generators: Generator class that we can extract a level string from
        @param solvers: n/a here; Solver class that can play a game
        @param kwargs: future proofing
        :return: True/False is this level a good level to use?
        """
        level = [row.split() for row in str(generators[0]).split('\n')[:-1]]
        graph = grid_graph(dim=(len(level), len(level[0])))
        agent_start_positions = []
        interesting_positions = []

        for j, row in enumerate(level):
            for i, char in enumerate(row):
                if char in self.game_schema.agent_chars:
                    agent_start_positions.append((i, j))
                if char == self.game_schema.wall_char:
                    graph.remove_node((i, j))
                if char in self.game_schema.interesting_chars:
                    interesting_positions.append((i, j))

        for agent in agent_start_positions:
            for interesting_position in interesting_positions:
                if not nx.has_path(graph, source=agent, target=interesting_position):
                    return False, {}

        return True, {}
