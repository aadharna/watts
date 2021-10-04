import unittest
import numpy as np
import torch
from torch import rand
from gym.spaces import Discrete, MultiDiscrete, Box
from models.categorical_action_sampler import ActionSampler


class TestActionSampler(unittest.TestCase):

    def test_multi_discrete_sampling(self):
        action_space = MultiDiscrete([2, 5])
        sampler = ActionSampler(action_space=action_space)
        # obs_space = Box(0.0, 255.0, (5, 5, 6), np.float64)
        # state = torch.FloatTensor([obs_space.sample()])
        # get NN and run the state through the NN to get real logits
        random_logits = rand(1, sum(action_space.nvec))
        actions, logp, entropy = sampler.sample(random_logits)
        print(actions)
        print(logp)
        print(entropy)

    def test_discrete_sampling(self):
        action_space = Discrete(5)
        sampler = ActionSampler(action_space=action_space)
        # obs_space = Box(0.0, 255.0, (5, 5, 6), np.float64)
        # state = torch.FloatTensor([obs_space.sample()])
        # get NN and run the state through the NN to get real logits
        random_logits = rand(32, action_space.n)
        actions, logp, entropy = sampler.sample(random_logits)
        print(actions)
        print(logp)
        print(entropy)

    def test_discrete_max_sampling(self):
        action_space = Discrete(5)
        sampler = ActionSampler(action_space=action_space)
        # obs_space = Box(0.0, 255.0, (5, 5, 6), np.float64)
        # state = torch.FloatTensor([obs_space.sample()])
        # get NN and run the state through the NN to get real logits
        logits = torch.FloatTensor([[0.01, 0.3, 0.29, 0.2, 0.2]])
        # rand(1, action_space.n)
        actions, logp, entropy = sampler.sample(logits, max=True)
        assert(actions == 1)

    def test_multi_discrete_max_sampling(self):
        action_space = MultiDiscrete([2, 5])
        sampler = ActionSampler(action_space=action_space)
        # obs_space = Box(0.0, 255.0, (5, 5, 6), np.float64)
        # state = torch.FloatTensor([obs_space.sample()])
        # get NN and run the state through the NN to get real logits
        random_logits = torch.FloatTensor([[0.6, 0.4, 0.2, 0.4, 0.2, 0.1, 0.1]])
        actions, logp, entropy = sampler.sample(random_logits, max=True)
        assert(actions[0][0] == 0, actions[0][1] == 1)

    def test_invalid_sampling(self):
        action_space = Box(-2.0, 2.0, (1,), np.float32)
        with self.assertRaises(ValueError):
            sampler = ActionSampler(action_space=action_space)
