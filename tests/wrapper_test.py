import os
import pytest
from griddly.util.rllib.environment.core import RLlibEnv
from watts.utils.gym_wrappers import \
        HierarchicalBuilderEnv, \
        AlignedReward, \
        SetLevelWithCallback, \
        Regret


def run_wrapper_test(game, wrapper):
    env_config = {
            'yaml_file': os.path.join('example_levels', f"{game}.yaml"),
    }
    base_env = RLlibEnv(env_config)
    env = wrapper(base_env, env_config)
    return env


def test_hierarchical_wrapper():
    env = run_wrapper_test('limited_zelda', HierarchicalBuilderEnv)
    assert env.builder_env
    assert env.env

def test_regret_wrapper():
    env = run_wrapper_test('maze', Regret)
    assert env.builder_env
    assert env.env

def test_aligned_wrapper():
    env = run_wrapper_test('foragers', AlignedReward)
    assert env.win is None
    assert env.steps == -1

def test_level_setting_callback_wrapper():
    env = run_wrapper_test('zelda', SetLevelWithCallback)
    level, data = env.create_level_fn()
    assert env.generation_data is None
    assert level is None
    assert data is None
