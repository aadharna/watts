import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelConfigDict
from griddly.util.rllib.torch.agents.common import layer_init

import torch
import torch.nn as nn


class TwoLayerFC(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self._num_objects = obs_space.shape[0]
        self._num_actions = num_outputs
        self.hidden_dim = model_config.get('custom_model_config', {}).get('hidden_size', 40)

        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self._num_objects, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self._num_actions)),
            nn.Tanh()
        )

        self.value_head = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, 1))
        )

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict['obs'].shape)
        x = input_dict['obs']
        self._last_batch_size = x.shape[0]

        embed = self.embedding(x)
        logits = self.policy_head(embed)
        value = self.value_head(embed)
        self._value = value.reshape(-1)
        return logits, state

    def value_function(self):
        return self._value


if __name__ == "__main__":
    build_info = {
        'obs_space': gym.spaces.Box([-np.inf for _ in range(24)],
                                    [np.inf for _ in range(24)], (24,)),
        'action_space': gym.spaces.Box([-1, -1, -1, -1], [1, 1, 1, 1], (4,)),
        'num_outputs': 4,
        'model_config': {'custom_model_config': {'hidden_size': 40}},
        'name': 'poet_fc'
    }

    model = TwoLayerFC(**build_info)
    print(model)
    obs = torch.FloatTensor([build_info['obs_space'].sample()])
    logits, _ = model.forward({'obs': obs}, [None], 1)
    print(logits)
    print(model.value_function())