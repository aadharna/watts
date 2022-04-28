from enum import Enum
from typing import Tuple
from itertools import product

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from watts.generators.base import BaseGenerator
from watts.models.PCGRL_network import PCGRLAdversarial


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
    """
    This generator no longer works with the breaking changes of making the
    NetworkFactory output full rllib::policy classes rather than ray pytorch NNs.
    This is on the docket to be fixed.

    This implements a recurrent neural network that sequentially generates a griddly game level.
    This is a bit too specific for my tastes to the zelda domain right now because of the Enum above.
    """
    id = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        BaseGenerator.__init__(self)
        self.network = PCGRLAdversarial(obs_space, action_space, num_outputs, model_config, name)
        self._num_objects = obs_space.shape[2]
        self.length = model_config.get('length', 15)
        self.width = model_config.get('width', 15)
        self.lvl_shape = (self.length, self.width)
        self.placements = model_config.get('placements', 50)
        self.max_sample = model_config.get('max_sampling', False)
        self.boundary_walls_height = list(product([0, self.length - 1], range(self.width)))
        self.boundary_walls_length = list(product(range(self.length), [0, self.width - 1]))
        self.id = PCGRLGenerator.id
        PCGRLGenerator.id += 1

    def mutate(self, **kwargs):
        param_vector = parameters_to_vector(self.network.parameters())
        # sample gaussian noise
        noise = kwargs.get('delta', torch.distributions.Normal(0, 1).sample((param_vector.shape[0], )))
        if not isinstance(noise, torch.Tensor):
            noise = torch.Tensor(noise)
        # add gaussian noise to the network params
        param_vector.add_(noise)

        # build new network
        new_generator = PCGRLGenerator(self.network.obs_space,
                                       self.network.action_space,
                                       self.network.num_outputs,
                                       self.network.model_config,
                                       self.network.name)
        # load in the noisy parameters
        vector_to_parameters(param_vector, new_generator.network.parameters())
        return new_generator

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
        # sampler = ActionSampler(self.network.action_space)

        blankMap = np.zeros((1, self._num_objects, length, width))
        level = torch.flatten(torch.FloatTensor(blankMap))
        actions = torch.zeros(self.placements, len(self.network.action_space.sample()) - 1)
        states = torch.zeros((self.placements, self._num_objects, length, width))
        rewards = torch.zeros((self.placements, 1))
        values = torch.zeros((self.placements, 1))
        masks = torch.zeros((self.placements, 1)) + 1
        logps = torch.zeros((self.placements, 1))
        h = self.network.get_initial_state()
        # do I put token based object adding back in? 
        # for i, TOKEN in enumerate(range(self.placements)):
        #     logits, h = self.network.forward_rnn(level, h, 1)
        #     torch_action, logp, entropy = sampler.sample(logits, max=self.max_sample)
        #     predicted = torch_action.cpu().numpy()
        #     x = int(predicted[0][0])
        #     y = int(predicted[0][1])
        #     tile = int(predicted[0][2])
        #     blankMap[0, tile, y, x] = 1
        #     level = torch.FloatTensor(blankMap)
        #     logps[i] = logp
        #     actions[i] = torch.FloatTensor([tile, y, x])
        #     states[i] = level.clone()
        #     level = torch.flatten(level)
        #     values[i] = self.network.value_function()
        #
        # for i, j in self.boundary_walls_length:
        #     # blankMap[0, self.game_schema.str_to_index['w'], i, j] = 1
        #     blankMap[0, Items.WALL.value, i, j] = 1
        # for i, j in self.boundary_walls_height:
        #     # blankMap[0, self.game_schema.str_to_index['w'], i, j] = 1
        #     blankMap[0, Items.WALL.value, i, j] = 1

        return torch.FloatTensor(blankMap), {"actions": actions,
                                             "states": states,
                                             "rewards": rewards,
                                             "values": values,
                                             "masks": masks,
                                             "logps": logps}

    def to_string(self, map):
        """

        @param map: numpy array of OHE map. The shape should be (1, num_objects, length, width)
        :return: level as string
        """
        level = ""
        map_shape = map.shape
        for i in range(map_shape[2]):
            for j in range(map_shape[3]):
                tile = torch.argmax(map[0, :, i, j]).item()
                level += Items.to_str(tile)
                # level += self.game_schema.index_to_str[tile]
                if j != map_shape[3] - 1:
                    level += " "
            level += "\n"
        return level


if __name__ == "__main__":
    from tests.test_structs import example_pcgrl_network_factory_build_info

    build_info = example_pcgrl_network_factory_build_info
    build_info['name'] = 'adversary'
    build_info['model_config'] = {'length': 15, 'width': 15, "placements": 75}
    print(build_info)

    generator = PCGRLGenerator(**build_info)

#     new_gen = generator.mutate()
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
