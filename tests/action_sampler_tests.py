import unittest
from torch import rand
from gym.spaces import Discrete, MultiDiscrete, Box
from models.categorical_action_sampler import ActionSampler


class TestActionSampler(unittest.TestCase):

    def test_multiDiscrete_sampling(self):
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

    def test_Discrete_sampling(self):
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
