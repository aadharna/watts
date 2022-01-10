import pytest
from copy import deepcopy

from watts.utils import Registrar
from watts.utils.loader import load_from_yaml


# load single agent policy
@pytest.fixture
def single_agent_policy_args():
    return load_from_yaml(fpath='sample_args/args.yaml')

@pytest.fixture
def multi_agent_policy_args():
    return load_from_yaml(fpath='sample_args/multi_policy_args.yaml')
    

# make sure single agent config builds correctly
@pytest.mark.parametrize(
        'optimization_alg',
        [
            'OpenAIES',
            'PPO',
            'MAML',
            'DDPG',
            'DQN',
            'SAC',
            'IMPALA'
        ]
    )
def test_single_agent_registrar(optimization_alg, single_agent_policy_args):
    """
    test that no multi-agent related config is loaded from file arguments
    """
    single_agent_policy_args.opt_algo = optimization_alg
    Registrar.id = 0
    registry = deepcopy(Registrar(file_args=single_agent_policy_args))
    trainer_config = registry.get_trainer_config
    assert trainer_config['multiagent']['policies'] == {}

# make sure multi agent config builds correctly
# with and without multiple policies
@pytest.mark.parametrize(
        'optimization_alg',
        [
            'OpenAIES',
            'PPO',
            'MAML',
            'DDPG',
            'DQN',
            'SAC',
            'IMPALA'
        ]
    )
def test_multi_policy_registrar(optimization_alg, multi_agent_policy_args):
    """
    test that multiple policies are handled correctly
    """
    multi_agent_policy_args.opt_algo = optimization_alg
    Registrar.id = 0
    registry = Registrar(file_args=multi_agent_policy_args)
    trainer_config = registry.get_trainer_config
    assert list(trainer_config['multiagent']['policies'].keys())[0] == 'harvester'

