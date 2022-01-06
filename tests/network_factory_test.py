import copy
import pytest

from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent
from watts import network_factory
from watts.models.AIIDE_network import AIIDEActor
from watts.models.PCGRL_network import PCGRLAdversarial

from tests.utils import test_structs


def run_network_factory_test(name: str, constructor, state_dict):
    build_info = copy.copy(test_structs.example_network_factory_build_info)
    if name == 'Adversarial_PCGRL':
        build_info = copy.copy(test_structs.example_pcgrl_network_factory_build_info)
    build_info['name'] = name
    nf = network_factory.NetworkFactory(name, build_info)
    assert nf.network_name == name
    assert nf.nn_build_info == build_info
    assert nf.constructor == constructor
    # Confirm actually making a network doesn't break things
    network = nf.make()(state_dict)
    assert network is not None


def test_aiide():
    run_network_factory_test(network_factory.aiide, AIIDEActor, test_structs.example_aiide_state_dict)

def test_conv():
    run_network_factory_test(network_factory.conv, SimpleConvAgent, test_structs.example_conv_state_dict)

def test_gap():
    run_network_factory_test(network_factory.gap, GAPAgent, test_structs.example_gap_state_dict)

def test_pcgrl():
    run_network_factory_test(network_factory.pcgrl, PCGRLAdversarial, test_structs.example_pcgrl_state_dict)

def test_no_state():
    run_network_factory_test(network_factory.aiide, AIIDEActor, {})

def test_invalid():
    with pytest.raises(ValueError):
        network_factory.NetworkFactory("foo", {})
