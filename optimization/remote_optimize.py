import ray


@ray.remote
def async_optimize_solver_on_env(trainer_constructor,
                                 trainer_config,
                                 registered_gym_name,
                                 level_string_monad,
                                 optimize_monad,
                                 **kwargs):
    """Run one step of optimization remotely!!

    :param trainer_constructor: constructor for algo to optimize wtih e.g. ppo.PPOTrainer for rllib to run optimization.
    :param trainer_config: config dict for e.g. PPO.
    :param registered_gym_name: name of env registered with ray via `env_register`
    :param level_string_monad:  callback to allow for dynamically created strings
    :param optimize_monad: function to run the evaluation. Solver.evaluate
    :param network_weights: torch state_dicts from Solver.get_weights()
    :return: dict of {optimized weights, result_dict}
    """
    return optimize_monad(trainer_constructor, trainer_config, registered_gym_name, level_string_monad,
                          **kwargs)
