import ray

from griddly.util.rllib.environment.core import RLlibEnv
from ray.tune.registry import register_env

from ray import tune
from ray.rllib.agents import ppo


@ray.remote
def optimize_agent_on_env(trainer_constructor,
                          trainer_config,
                          registered_gym_name,
                          level_string_monad,
                          actor_critic_weights,
                          **kwargs):
    """Run one step of optimization remotely!!

    :param trainer_constructor: constructor for algo to optimize wtih e.g. ppo.PPOTrainer for rllib to run optimization.
    :param trainer_config: config dict for e.g. PPO.
    :param registered_gym_name: name of env registered with ray via `env_register`
    :param level_string_monad:  callback to allow for dynamically created strings
    :param actor_critic_weights: torch state_dict
    :return: dict of {optimized weights, result_dict}
    """

    # todo same as rollout.py
    # todo will probably have to change this to first instantiate a generator model
    # and then query it for the levels.
    #  That will allow something like PAIRED to function?
    trainer_config['env_config']['level_string'] = level_string_monad()
    trainer = trainer_constructor(config=trainer_config, env=registered_gym_name)
    trainer.get_policy().model.load_state_dict(actor_critic_weights)
    result = trainer.train()

    return {'weights': trainer.get_policy().model.state_dict(),
            "result_dict": result,
            'pair_id': kwargs.get('pair_id', 0)
            }


if __name__ == "__main__":
    import os
    import sys
    import ray
    from ray.tune.registry import register_env

    from utils.loader import load_from_yaml
    from utils.register import Registrar
    from utils.gym_wrappers import AlignedReward
    from models.AIIDE_network import AIIDEActor
    from network_factory import NetworkFactory
    from gym_factory import GridGameFactory

    from pprint import pprint

    os.chdir('..')
    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init()

    args_file = os.path.join('args.yaml')
    args = load_from_yaml(args_file)

    registry = Registrar(file_args=args)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info)
    gym_factory = GridGameFactory(registry.env_name, [AlignedReward])

    init_net = network_factory.make()()
    print(init_net)
    init_weights = init_net.state_dict()

    # register_env(registry.name, gym_factory.make())
    # print(registry.name)
    # pprint(registry.trainer_config['env_config'])
    # pprint(registry.trainer_config['model'])

    try:
        opt_ref = optimize_agent_on_env.remote(registry.trainer_constr,
                                               registry.trainer_config,
                                               registry.name,
                                               lambda: None,
                                               init_weights,
                                               pair_id=0)
        return_dict = ray.get(opt_ref)
        # print(return_dict)
    except ray.tune.error.TuneError as e:
        ray.shutdown()



    ray.shutdown()
