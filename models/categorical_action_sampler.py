import gym
import numpy as np
import torch
from torch.distributions import Categorical


class ActionSampler:

    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space

        if isinstance(action_space, gym.spaces.Discrete):
            self._action_space_shape = [action_space.n]
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self._action_space_shape = action_space.nvec
        else:
            raise ValueError(f"{action_space} is an unsupported action space")

        self._num_action_logits = np.sum(self._action_space_shape)
        self._num_action_parts = len(self._action_space_shape)

    def sample(self, logits):
        """

        :param logits:
        :return:
        """
        batch_size = logits.shape[0]
        actions = torch.zeros([batch_size, self._num_action_parts])
        logps = torch.zeros([batch_size])
        entropies = torch.zeros([self._num_action_parts])

        split = logits.split(tuple(self._action_space_shape), dim=1)

        offset = 0
        for i, (subset, subset_size) in enumerate(zip(split, self._action_space_shape)):

            dist = Categorical(logits=subset)
            sampled = dist.sample()
            logp = dist.log_prob(sampled)
            logps += logp
            entropies[i] = dist.entropy().mean()

            actions[:, i] = sampled
            offset += subset_size

        return actions, logps, torch.sum(entropies)


if __name__ == '__main__':
    import os
    from utils.loader import load_from_yaml
    from utils.register import Registrar
    from network_factory import NetworkFactory

    os.chdir('..')

    args = load_from_yaml("args.yaml")
    reg = Registrar(file_args=args)
    nf = NetworkFactory(network_name=reg.network_name, nn_build_info=reg.get_nn_build_info)
    nn = nf.make()({})
    states = []
    for _ in range(5):
        states.append(reg.obs_space.sample())
    states = torch.FloatTensor(states)
    logits, _ = nn({'obs': states}, None, None)
    sampler = ActionSampler(reg.act_space)
    actions, logp, entropy = sampler.sample(logits)
    print(actions)
    print(logp)
    print(entropy)
