# import copy
# import unittest
#
# from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
# from griddly.util.rllib.torch.agents.global_average_pooling_agent import GAPAgent
# from watts import network_factory
# from watts.models.AIIDE_network import AIIDEActor
# from watts.models.PCGRL_network import PCGRLAdversarial
# from watts.utils.register import get_default_trainer_config_constructor_and_policy_fn
# from tests import test_structs
#
#
# def prep_config(trainer_config, network_name):
#     trainer_config["model"] = {
#         'custom_model': network_name,
#         'custom_model_config': {},
#         'max_seq_len': 20  # gotten from the ppo default config
#     }
#     trainer_config['framework'] = 'torch'
#
#
# def run_network_factory_test(name: str, constructor, state_dict):
#     config, _, get_policy_fn = get_default_trainer_config_constructor_and_policy_fn('PPO')
#     config = prep_config(config, name)
#     policy_class = get_policy_fn(config)
#     build_info = copy.copy(test_structs.example_network_factory_build_info)
#     if name == 'Adversarial_PCGRL':
#         build_info = copy.copy(test_structs.example_pcgrl_network_factory_build_info)
#     build_info['config'] = config
#     nf = network_factory.NetworkFactory(name, build_info, policy_class)
#     assert nf.network_name == name
#     assert nf.nn_build_info == build_info
#     assert nf.constructor == constructor
#     # Confirm actually making a network doesn't break things
#     network = nf.make()(state_dict)
#     assert network is not None
#
#
# class TestNetworkFactory(unittest.TestCase):
#
#     def test_aiide(self):
#         run_network_factory_test(network_factory.aiide, AIIDEActor, test_structs.example_aiide_state_dict)
#
#     def test_conv(self):
#         run_network_factory_test(network_factory.conv, SimpleConvAgent, test_structs.example_conv_state_dict)
#
#     def test_gap(self):
#         run_network_factory_test(network_factory.gap, GAPAgent, test_structs.example_gap_state_dict)
#
#     def test_pcgrl(self):
#         run_network_factory_test(network_factory.pcgrl, PCGRLAdversarial, test_structs.example_pcgrl_state_dict)
#
#     def test_no_state(self):
#         run_network_factory_test(network_factory.aiide, AIIDEActor, {})
#
#     def test_invalid(self):
#         with self.assertRaises(ValueError):
#             network_factory.NetworkFactory("foo", {})
