import ray

from griddly.util.rllib.wrappers.core import RLlibEnv
from ray.tune.registry import register_env

from ray import tune
from ray.rllib.agents import ppo


@ray.remote
def optimize_agent_on_env(gym_factory_monad, level_string, actor_critic_weights, trainer_config):
    """Run one step of optimization remotely!!

    :param gym_factory_monad: registry function wrapper
    :param level_string: level as a string (this is temporary and will
                            also become a callback to allow for dynamically created strings)
    :param actor_critic_weights: torch state_dict
    :param trainer_config: config dict for e.g. PPO.
    :return: dict of {optimized weights, result_dict}
    """

    gym_factory_monad()
    trainer_config['env_config']['level_string'] = level_string
    trainer = ppo.PPOTrainer(config=trainer_config, env=trainer_config['env'])
    trainer.get_policy().model.load_state_dict(actor_critic_weights)
    result = trainer.train()

    return {'weights': trainer.get_policy().model.state_dict(), "result_dict": result}


if __name__ == "__main__":
    import os
    import ray
    import gym
    import griddly
    from ray.tune.registry import register_env
    from griddly.util.rllib.wrappers.core import RLlibEnv
    from utils.loader import load_from_yaml
    from utils.register import Registrar
    from network_factory import NetworkFactory
    from gym_factory import GridGameFactory

    os.chdir('..')
    ray.init()

    register_env('foo', RLlibEnv)

    args_file = os.path.join('args.yaml')
    args = load_from_yaml(args_file)

    registry = Registrar(file_args=args)
    network_factory = NetworkFactory(registrar=registry)
    gym_factory = GridGameFactory(args, [], registry)

    init_net = network_factory.make()()
    init_weights = init_net.state_dict()

    opt_ref = optimize_agent_on_env.remote(gym_factory.register(), None, init_weights, registry.trainer_config)

    return_dict = ray.get(opt_ref)
    print(return_dict)

    ray.shutdown()
