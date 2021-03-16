from utils.loader import load_from_yaml

from generators.base import BaseGenerator
from ray.rllib.agents.trainer import Trainer

class Pair:
    id = 0

    def __init__(self, file_args, agent: Trainer, generator: BaseGenerator):

        self.args = file_args

        self.agent = agent
        self.generator = generator

        self.id = Pair.id
        Pair.id += 1

    def __str__(self):
        return str(self.generator)

    # def update_agent(self, new_weights):
    #     self.agent.get_policy().set_weights(new_weights)

    # def _evaluate(self) -> float:
    #     """
    #     The generator at this time has a level embedded in it already.
    #     Evaluate the attached NN in this defined map.
    #     :return: float
    #     """
    #     resultDict = self.agent._evaluate()
    #     return resultDict['episode_reward_mean']
    #
    # def _train(self):
    #     """
    #
    #     :return:
    #     """
    #     ptr = self.agent.train()
    #     return ptr


if __name__ == "__main__":
    pass
