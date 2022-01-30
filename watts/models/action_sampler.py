import gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class ActionSampler:

    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space
        self.is_discrete = True

        if isinstance(action_space, gym.spaces.Discrete):
            self._action_space_shape = [action_space.n]
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self._action_space_shape = action_space.nvec
        elif isinstance(action_space, gym.spaces.Box):
            # we need logits for a mu vector
            # we need logits for a std vector
            self._action_space_shape = [action_space.shape[0], action_space.shape[0]]
            self.is_discrete = False
        else:
            raise ValueError(f"{action_space} is an unsupported action space")

        self._num_action_logits = np.sum(self._action_space_shape)
        self._num_action_parts = len(self._action_space_shape)

    def sample(self, logits, max=False):
        """

        :param logits:
        :return:
        """
        batch_size = logits.shape[0]

        if self.is_discrete:
            actions = torch.zeros([batch_size, self._num_action_parts])
            logps = torch.zeros([batch_size])
            entropies = torch.zeros([self._num_action_parts])
            split = logits.split(tuple(self._action_space_shape), dim=1)
            offset = 0
            for i, (subset, subset_size) in enumerate(zip(split, self._action_space_shape)):

                dist = Categorical(logits=subset)
                sampled = dist.sample() if not max else torch.argmax(subset)
                logp = dist.log_prob(sampled)
                logps += logp
                entropies[i] = dist.entropy().mean()

                actions[:, i] = sampled
                offset += subset_size

        else:
            # each dim is an independent gaussian therefore we need the probability of each dim/sample
            actions = torch.zeros([batch_size, self.action_space.shape[0]])
            # the above is why thtis is batchsize x dim
            logps = torch.zeros([batch_size, self.action_space.shape[0]])
            entropies = torch.zeros([1])

            try:
                mu, sigma = logits.split(tuple(self._action_space_shape), dim=1)
            except RuntimeError as e:
                mu = logits
                sigma = torch.zeros(logits.shape)
            sigma = sigma.exp().expand_as(mu)
            dist = Normal(mu, sigma)
            sampled = dist.sample()
            logp = dist.log_prob(sampled)
            logps += logp
            entropies[0] = dist.entropy().mean()
            actions[0] = sampled

        return actions, logps, torch.sum(entropies)


if __name__ == '__main__':
    act_space = gym.spaces.Box(np.array([-1, -1, -1, -1]),
                               np.array([1, 1, 1, 1]), (4,))
    sampler = ActionSampler(act_space)
    action, logs, ent  = sampler.sample(torch.rand(1, 8))
    print(action)