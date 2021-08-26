from gym.spaces import Box, MultiDiscrete
import numpy as np
from torch import rand

example_network_factory_build_info = {
    'action_space': MultiDiscrete([2, 5]),
    'obs_space': Box(0.0, 255.0, (5, 5, 6), np.float64),
    'model_config': {},
    'num_outputs': 7,
    # 'name' needs to be added based on which network is being run
}

example_pcgrl_network_factory_build_info = {
    'action_space': MultiDiscrete([15, 15, 6, 2]),
    'obs_space': Box(0.0, 255.0, (15, 15, 6), np.float64),
    'model_config': {'length': 15, 'width': 15, 'placements': 50},
    'num_outputs': sum([15, 15, 6, 2]),
    # 'name' needs to be added based on which network is being run
}

example_aiide_state_dict = {
    "embedding.0.weight": rand((8, 6, 1, 1)),
    "embedding.0.bias": rand(8),
    "embedding.2.weight": rand((32, 8, 2, 2)),
    "embedding.2.bias": rand(32),
    "embedding.5.weight": rand((128, 512)),
    "embedding.5.bias": rand(128),
    "policy_head.0.weight": rand((7, 128)),
    "policy_head.0.bias": rand(7),
    "value_head.0.weight": rand((1, 128)),
    "value_head.0.bias": rand(1)
}

example_conv_state_dict = {
    "network.0.weight": rand((32, 6, 3, 3)),
    "network.0.bias": rand(32),
    "network.2.weight": rand((64, 32, 3, 3)),
    "network.2.bias": rand(64),
    "network.5.weight": rand((1024, 1600)),
    "network.5.bias": rand(1024),
    "network.7.weight": rand((512, 1024)),
    "network.7.bias": rand(512),
    "_actor_head.0.weight": rand((256, 512)),
    "_actor_head.0.bias": rand(256),
    "_actor_head.2.weight": rand((7, 256)),
    "_actor_head.2.bias": rand(7),
    "_critic_head.0.weight": rand((1, 512)),
    "_critic_head.0.bias": rand(1)
}

example_gap_state_dict = {
    "network.0.weight": rand((32, 6, 3, 3)),
    "network.0.bias": rand(32),
    "network.2.weight": rand((64, 32, 3, 3)),
    "network.2.bias": rand(64),
    "network.5.weight": rand((1024, 2048)),
    "network.5.bias": rand(1024),
    "network.7.weight": rand((512, 1024)),
    "network.7.bias": rand(512),
    "_actor_head.0.weight": rand((256, 512)),
    "_actor_head.0.bias": rand(256),
    "_actor_head.2.weight": rand((7, 256)),
    "_actor_head.2.bias": rand(7),
    "_critic_head.0.weight": rand((1, 512)),
    "_critic_head.0.bias": rand(1)
}

example_pcgrl_state_dict = {
    "conv.weight": rand((16, 4, 3, 3)),
    "conv.bias": rand(16),
    "rnn.weight_ih": rand((8112, 2704)),
    "rnn.weight_hh": rand((8112, 2704)),
    "rnn.bias_ih": rand(8112),
    "rnn.bias_hh": rand(8112),
    "fc.weight": rand((38, 2704)),
    "fc.bias": rand(38),
    "val.weight": rand((1, 2704)),
    "val.bias": rand(1)
}