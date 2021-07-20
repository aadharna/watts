import ray


@ray.remote
def async_evaluate_agent_on_level(gym_factory_monad,
                                  rllib_env_config,
                                  level_string_monad,
                                  evaluate_monad,
                                  solver_id,
                                  generator_id):
    """

    :param gym_factory_monad: func to create an env
    :param rllib_env_config: dictionary of necessary information to make the env
    :param level_string_monad: what level do you want to load into the game defined by the above gdy file
    :param network_factory_monad: Factory_make function for NN
    :param evaluate_monad: function that does the evaluation -- member function of a Solver class
    :param network_weights: weights for your actor-critic network
    :param solver_id: id of solver being evaluated
    :param generator_id: id of generator the solver is being evaluated in
    :return: dict of rollout information
    """

    # todo will probably have to change this to first instantiate a generator model
    # and then query it for the levels.
    #  That will allow something like PAIRED to function.
    rllib_env_config['level_string'], _ = level_string_monad()
    env = gym_factory_monad(rllib_env_config)

    result_dictionary = evaluate_monad(env)
    result_dictionary.update({'solver_id': solver_id, 'generator_id': generator_id})

    env.close()
    env.game.release()

    return result_dictionary
