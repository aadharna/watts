import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import ModelConfigDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork, ModelV2
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class PCGRLAdversarial(RecurrentNetwork, nn.Module):
    """generates a map tile by tile"""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self._num_objects = obs_space.shape[2]
        self.cell_size = model_config.get('cell_size', 2704)

        self.conv = nn.Conv2d(in_channels=self._num_objects, out_channels=16, kernel_size=3)
        self.flat = nn.Flatten()
        self.rnn = nn.GRUCell(input_size=self.cell_size, hidden_size=self.cell_size)
        self.fc = nn.Linear(in_features=self.cell_size, out_features=num_outputs)

        self.val = nn.Linear(self.cell_size, 1)

    @override(RecurrentNetwork)
    def forward_rnn(self, x, state, seq_lens):
        x = x.reshape(-1, self._num_objects, self.obs_space.shape[0], self.obs_space.shape[1])
        x = self.conv(x)
        x = self.flat(x)
        h_in = state[0].reshape(-1, self.cell_size)
        # print(f"seq_len {seq_lens}")
        # print(f"x.shape {x.shape}, h_in.shape {h_in.shape}")
        if not h_in.shape[0] == x.shape[0]:
            missing_h = self.conv.weight.new(x.shape[0] - h_in.shape[0], h_in.shape[1]).zero_()
            h_in = torch.vstack((h_in, missing_h))
        h = self.rnn(x, h_in)
        self.value = self.val(h)
        logits = self.fc(h)
        # print(f"h.shape {h.shape}, logits.shape {logits.shape}, self.value.shape {self.value.shape}")
        # print('_'*60)
        return logits, [h]

    @override(ModelV2)
    def value_function(self):
        return self.value.squeeze(1)

    @override(ModelV2)
    def get_initial_state(self):  # -> List[np.ndarray]:
        """Get the initial recurrent state values for the model.

        Returns:
            List[np.ndarray]: List of np.array objects containing the initial
                hidden state of an RNN, if applicable.

        """
        h = [self.conv.weight.new(
                1, self.cell_size).zero_()]
        return h


if __name__ == "__main__":
    from gym.spaces import Box, Discrete, MultiDiscrete
    from models.categorical_action_sampler import ActionSampler

    example_network_factory_build_info = {
        'action_space': MultiDiscrete([15, 15,  6,  2]),
        'obs_space': Box(0.0, 255.0, (15, 15, 4), np.float64),
        'model_config': {'length': 15, 'width': 15, 'placements': 50},
        'num_outputs': sum([15, 15,  6,  2]),
        'name': 'pcgrl'
    }
    adversary = PCGRLAdversarial(**example_network_factory_build_info)
    print(adversary)
    blankMap = np.zeros((1, 4, 15, 15))
    level = torch.FloatTensor(blankMap)
    h = adversary.get_initial_state()
    logits, h = adversary.forward_rnn(level, h, 1)
    sampler = ActionSampler(example_network_factory_build_info['action_space'])
    action, logp, entropies = sampler.sample(logits=logits)
    print(action)
