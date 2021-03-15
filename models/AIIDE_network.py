import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelConfigDict
from griddly.util.rllib.torch.agents.common import layer_init


class AIIDEActor(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self._num_objects = obs_space.shape[2]
        self._num_actions = num_outputs

        self.embedding = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=self._num_objects, out_channels=8, kernel_size=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(640, 128)),  # was 512 previously
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(128, num_outputs))
        )

        self.value_head = nn.Sequential(
            layer_init(nn.Linear(128, 1))
        )

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict['obs'].shape)
        x = input_dict['obs'].permute(0, 3, 1, 2)
        self._last_batch_size = x.shape[0]

        embed = self.embedding(x)
        logits = self.policy_head(embed)
        value = self.value_head(embed)
        self._value = value.reshape(-1)
        return logits, state

    def value_function(self):
        return self._value


if __name__ == "__main__":
    import numpy as np
    import gym
    import os
    import griddly
    from griddly import gd

    from utils.loader import load_from_yaml
    from utils.register import register_env_with_griddly
    from gym_factory import GridGameFactory

    os.chdir('..')
    args = load_from_yaml(os.path.join('args.yaml'))

    name, nActions, actSpace, obsSpace, observer = register_env_with_griddly(file_args=args)

    gameFactory = GridGameFactory(args, name, nActions, actSpace, obsSpace, observer, env_wrappers=[])
    env = gameFactory.make()()
    state = env.reset()
    print(env.observation_space.shape)

    #shape should = 640 in this test, but 512 in rllib. I don't know why.

    network = AIIDEActor(obs_space=env.observation_space, action_space=env.action_space,
                         num_outputs=nActions, model_config={}, name='AIIDE_PINSKY_MODEL')

    foo, _ = network({'obs': torch.FloatTensor([state])}, [0], 1)
    print(foo)
