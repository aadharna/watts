import os
import unittest

from game.GameSchema import GameSchema


class TestGameSchema(unittest.TestCase):

    def test_endless_butterflies(self):
        endless_butterflies_gdy = os.path.join('example_levels', 'endless_butterflies.yaml')
        game_schema = GameSchema(endless_butterflies_gdy)
        self.assertEqual({'A'}, game_schema.agent_chars)
        self.assertEqual('w', game_schema.wall_char)
        self.assertEqual({'b', 'S', 'c'}, game_schema.interesting_chars)

    def test_foragers(self):
        foragers_gdy = os.path.join('example_levels', 'foragers.yaml')
        game_schema = GameSchema(foragers_gdy)
        self.assertEqual({'f1', 'f2', 'f3', 'f4'}, game_schema.agent_chars)
        self.assertEqual('W', game_schema.wall_char)
        self.assertEqual({'g', 'r', 'b'}, game_schema.interesting_chars)

    def test_limited_zelda(self):
        limited_zelda_gdy = os.path.join('example_levels', 'limited_zelda.yaml')
        game_schema = GameSchema(limited_zelda_gdy)
        self.assertEqual({'A'}, game_schema.agent_chars)
        self.assertEqual('w', game_schema.wall_char)
        self.assertEqual({'x', '+', 'g', 'e'}, game_schema.interesting_chars)

    def test_maze(self):
        maze_gdy = os.path.join('example_levels', 'maze.yaml')
        game_schema = GameSchema(maze_gdy)
        self.assertEqual({'A'}, game_schema.agent_chars)
        self.assertEqual('w', game_schema.wall_char)
        self.assertEqual({'x', 't'}, game_schema.interesting_chars)

    def test_zelda(self):
        zelda_gdy = os.path.join('example_levels', 'zelda.yaml')
        game_schema = GameSchema(zelda_gdy)
        self.assertEqual({'A'}, game_schema.agent_chars)
        self.assertEqual('w', game_schema.wall_char)
        self.assertEqual({'x', '+', 'g', 'e'}, game_schema.interesting_chars)
