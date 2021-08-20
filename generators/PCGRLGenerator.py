from enum import Enum
from itertools import product
import numpy as np
from typing import Tuple
import torch

from generators.base import BaseGenerator
from models.PCGRL_network import PCGRLAdversarial
from models.categorical_action_sampler import ActionSampler


class Items(Enum):

    FLOOR  = 0
    DOOR   = 1
    KEY    = 2
    AVATAR = 3
    ENEMY  = 4
    WALL   = 5

    @staticmethod
    def to_str(tile):
        if tile == Items.FLOOR.value:
            return '.'
        elif tile == Items.DOOR.value:
            return 'g'
        elif tile == Items.KEY.value:
            return '+'
        elif tile == Items.AVATAR.value:
            return 'A'
        elif tile == Items.ENEMY.value:
            return 'e'
        elif tile == Items.WALL.value:
            return 'w'
        else:
            raise ValueError(f"unrecognized tile choice of {tile}")


class PCGRLGenerator(BaseGenerator):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        BaseGenerator.__init__(self)
        self.network = PCGRLAdversarial(obs_space, action_space, num_outputs, model_config, name)
        self._num_objects = obs_space.shape[2]
        self.length = model_config.get('length', 15)
        self.width = model_config.get('width', 15)
        self.lvl_shape = (self.length, self.width)
        self.placements = model_config.get('placements', 50)
        self.boundary_walls_height = list(product([0, self.length - 1], range(self.width)))
        self.boundary_walls_length = list(product(range(self.length), [0, self.width - 1]))

    def mutate(self, **kwargs):
        return self

    def update(self, level, **kwargs):
        pass

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            map, data = self.generate()
            return self.to_string(map), data
        return _generate

    @property
    def shape(self):
        return self.lvl_shape

    def __str__(self):
        """This function is necessary for our generators. This function will take whatever
        the internal representation of our "levels" are and turn it into a string that we
        can then use to set the simulator to in Griddly.

        :return:
        """
        map, _ = self.generate()
        return self.to_string(map)

    def generate(self):
        length = self.length
        width = self.width
        sampler = ActionSampler(self.network.action_space)

        blankMap = np.zeros((1, self._num_objects, length, width))
        level = torch.FloatTensor(blankMap)
        actions = torch.zeros(self.placements, len(self.network.action_space.sample()))
        states = torch.zeros((self.placements, self._num_objects, length, width))
        rewards = torch.zeros((self.placements, 1))
        values = torch.zeros((self.placements, 1))
        masks = torch.zeros((self.placements, 1)) + 1
        logps = torch.zeros((self.placements, 1))
        h = torch.zeros((1, 2704))
        tokenList = [Items.AVATAR, Items.DOOR, Items.KEY] + [Items.WALL for _ in range(self.placements - 3)]
        for i, TOKEN in enumerate(tokenList):
            logits, h = self.network.forward_rnn({'obs': level}, h, 1)
            torch_action, logp, entropy = sampler.sample(logits)
            predicted = torch_action.cpu().item()
            #
            # The mutable part of the map is a 13x13 subgrid of the 15x15 space
            # (0, 0) -> (1, 1); (i, j) -> (i+1, j+1)
            # Therefore, to ensure that the blocks get placed
            # into the 13x13 middle of the 15x15 grid, we use a 13x13 grid and shift the indices by +1, +1
            y = int(predicted % (length - 2)) + 1
            x = int(predicted // (width - 2)) + 1
            blankMap[0, TOKEN.value, y, x] = 1
            level = torch.FloatTensor(blankMap)
            logps[i] = logp
            actions[i] = torch.FloatTensor([TOKEN.value, y, x])
            states[i] = level.clone()
            values[i] = self.network.value_function()

        for i, j in self.boundary_walls_length:
            blankMap[0, Items.WALL.value, i, j] = 1
        for i, j in self.boundary_walls_height:
            blankMap[0, Items.WALL.value, i, j] = 1

        return torch.FloatTensor(blankMap), {"actions": actions,
                                             "states": states,
                                             "rewards": rewards,
                                             "values": values,
                                             "masks": masks,
                                             "logps": logps}

    def to_string(self, map):
        """

        :param map: numpy array of OHE map. The shape should be (1, num_objects, length, width)
        :return: level as string
        """
        level = ""
        map_shape = map.shape
        for i in range(map_shape[2]):
            for j in range(map_shape[3]):
                tile = torch.argmax(map[0, :, i, j]).item()
                level += Items.to_str(tile)
            level += "\n"
        return level


# if __name__ == "__main__":
#     import gym
#     from utils.returns import compute_gae
#     from tests.test_structs import example_network_factory_build_info
#     import torch.optim as optim
#
#     build_info = example_network_factory_build_info
#     build_info['action_space'] = gym.spaces.Discrete(169)
#     build_info['num_outputs'] = 169
#     build_info['name'] = 'adversary'
#     build_info['model_config'] = {'length': 15, 'width': 15, "placements": 75}
#
#     generator = PCGRLGenerator(**build_info)
#     optimizer = optim.Adam(generator.network.parameters(), lr=0.003)
#     mazes = []
#
#     for _ in range(15):
#         maze, info_dict = generator.generate()
#         mazes.append(maze)
#         returns = compute_gae(generator.network.value_function(), info_dict['rewards'], info_dict['masks'], info_dict['values'])
#         returns = torch.cat(returns)
#         advantage = returns - info_dict['values']
#         loss = (info_dict['logps'] * advantage).mean()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
