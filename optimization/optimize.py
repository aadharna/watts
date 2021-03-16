
from griddly.util.rllib.wrappers.core import RLlibEnv
from ray.tune.registry import register_env

from ray import tune
from ray.rllib.agents import ppo


# @ray.remote
def optimize_agent_on_env(gym_factory_monad, network_factory_monad, level_string, actor_critic_weights, agent_config):
    """

    :param gym_factory_monad:
    :param network_factory_monad:
    :param level_string:
    :param actor_critic_weights:
    :return:
    """

    trainer = ppo.PPOTrainer(config=agent_config, env=gym_factory_monad)
    result = trainer.train()

    return {'weights': trainer.get_policy().get_weights(), "result_dict": result}
