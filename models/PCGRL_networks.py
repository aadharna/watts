from enum import Enum

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym.spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelConfigDict


class PCGRLAdversarial(TorchModelV2, nn.Module):
    """generates a map tile by tile"""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.conv = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.flat = nn.Flatten()
        self.rnn = nn.LSTMCell(input_size=1936, hidden_size=32)
        self.fc = nn.Linear(in_features=32, out_features=num_outputs)
        self.px = torch.zeros((1, 32))
        self.ph = torch.zeros((1, 32))

        self.val = nn.Linear(32, 1)

    def forward(self, x, state, seq_lens):
        x = self.conv(x)
        x = self.flat(x)
        self.px, self.ph = self.rnn(x, (self.px, self.ph))
        x = self.px

        self.value = self.val(x)
        logits = self.fc(x)
        foo, predicted = torch.max(logits.squeeze(), 0)
        return predicted, state

    def value_function(self):
        return self.value

    def genearate(self, length, width):
        blankMap = np.zeros((1, 6, length, width))
        # todo fix selection and placement.
        tokenList = [Items.AVATAR, Items.DOOR, Items.KEY] + [Items.WALL for _ in range((length - 2) * (width - 2) - 3)]
        for i, ITEM in zip(range((length - 2) * (width - 2)), tokenList):
            spot, _ = self.forward(torch.FloatTensor(blankMap), [None], None)
            y = spot % (length - 2)
            x = spot // (width - 2)
            # We need to make sure that this assignment doesn't 
            # actually assign anything to the boundary of the length x width level object.
            #   A 15 x 15 level has 13 x 13 mutable tiles. 
            # Also, number of tiles to be placed should be a param.
            blankMap[0, ITEM.value, y, x] = 1

        # todo: add in:
        #   ITEM, mapping to string, map_size_inference,
        #         inferrable out_features, sampling from categorical in addition to max.
        self.map = blankMap
        return self.map

    def to_string(self):
        # todo: make the One-Hot level into a string to pass it to Griddly.
        return


class Items(Enum):

    AVATAR = 0
    DOOR   = 1
    KEY    = 2
    FLOOR  = 3
    ENEMY  = 4
    WALL   = 5

    def choose(self):
        pass


if __name__ == "__main__":
    pass
    # adversary = PCGRLAdversarial(...)
    # level = adversary.genearate(length=15, width=15)
    #
    # level = level.squeeze()
    # level[4, 5, 0] = 235
    #
    # foo = np.argmax(level, axis=0)
    # print(foo)
