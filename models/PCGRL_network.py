
import numpy as np

import torch
import torch.nn as nn

import gym
from ray.rllib.models.torch.torch_modelv2 import ModelConfigDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork


class PCGRLAdversarial(RecurrentNetwork, nn.Module):
    """generates a map tile by tile"""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)
        nn.Module.__init__(self)

        self._num_objects = obs_space.shape[2]

        self.conv = nn.Conv2d(in_channels=self._num_objects, out_channels=16, kernel_size=3)
        self.flat = nn.Flatten()
        self.rnn = nn.GRUCell(input_size=2704, hidden_size=2704)
        self.fc = nn.Linear(in_features=2704, out_features=num_outputs)

        self.val = nn.Linear(2704, 1)

    def forward_rnn(self, x, state, seq_lens):
        x = x['obs']
        x = self.conv(x)
        x = self.flat(x)
        h_in = state[0].reshape(-1, 2704)
        h = self.rnn(x, h_in)

        self.value = self.val(x)
        logits = self.fc(x)
        return logits, [h]

    def value_function(self):
        return self.value


if __name__ == "__main__":
    from gym.spaces import Box, Discrete

    example_network_factory_build_info = {
        'action_space': Discrete(169),
        'obs_space': Box(0.0, 255.0, (5, 5, 6), np.float64),
        'model_config': {'length': 15, 'width': 15, 'placements': 50},
        'num_outputs': 169,
        'name': 'pcgrl'
    }
    adversary = PCGRLAdversarial(**example_network_factory_build_info)
    print(adversary)
    blankMap = np.zeros((1, 6, 15, 15))
    level = torch.FloatTensor(blankMap)
    h = torch.zeros((1, 2704))
    logits, h = adversary.forward_rnn({'obs': level}, [h], 1)
