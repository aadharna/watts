import os
import pytest

from watts.game.GameSchema import GameSchema


def test_endless_butterflies():
    endless_butterflies_gdy = os.path.join('example_levels', 'endless_butterflies.yaml')
    game_schema = GameSchema(endless_butterflies_gdy)
    assert({'A'} == game_schema.agent_chars)
    assert('w' == game_schema.wall_char)
    assert({'b', 'S', 'c'} == game_schema.interesting_chars)

def test_foragers():
    foragers_gdy = os.path.join('example_levels', 'foragers.yaml')
    game_schema = GameSchema(foragers_gdy)
    assert({'f1', 'f2', 'f3', 'f4'} == game_schema.agent_chars)
    assert('W' == game_schema.wall_char)
    assert({'g', 'r', 'b'} == game_schema.interesting_chars)

def test_limited_zelda():
    limited_zelda_gdy = os.path.join('example_levels', 'limited_zelda.yaml')
    game_schema = GameSchema(limited_zelda_gdy)
    assert({'A'} ==  game_schema.agent_chars)
    assert('w' == game_schema.wall_char)
    assert({'x', '+', 'g', 'e'} == game_schema.interesting_chars)

def test_maze():
    maze_gdy = os.path.join('example_levels', 'maze.yaml')
    game_schema = GameSchema(maze_gdy)
    assert({'A'} == game_schema.agent_chars)
    assert('w' == game_schema.wall_char)
    assert({'x', 't'} == game_schema.interesting_chars)

def test_zelda():
    zelda_gdy = os.path.join('example_levels', 'zelda.yaml')
    game_schema = GameSchema(zelda_gdy)
    assert({'A'} == game_schema.agent_chars)
    assert('w' == game_schema.wall_char)
    assert({'x', '+', 'g', 'e'} == game_schema.interesting_chars)
