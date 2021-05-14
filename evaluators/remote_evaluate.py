from evaluators.evaluate import evaluate
import ray


@ray.remote
def async_evaluate_agent_on_level(gym_factory_monad,
                                  rllib_env_config,
                                  level_string_monad,
                                  network_factory_monad,
                                  actor_critic_weights,
                                  solver_id,
                                  generator_id):
    """

    :param gym_factory_monad: func to create an env
    :param rllib_env_config: dictionary of necessary information to make the env
    :param level_string_monad: what level do you want to load into the game defined by the above gdy file
    :param network_factory_monad: Factory_make function for NN
    :param actor_critic_weights: weights for your actor-critic network
    :return:
    """

    actor = network_factory_monad(actor_critic_weights)
    # todo will probably have to change this to first instantiate a generator model
    # and then query it for the levels.
    #  That will allow something like PAIRED to function.
    rllib_env_config['level_string'] = level_string_monad()
    env = gym_factory_monad(rllib_env_config)

    info, rewards, win = evaluate(actor, env)

    env.close()
    env.game.release()

    return {'score': rewards, 'win': win == "Win", 'info': info,
            'solver_id': solver_id, 'generator_id': generator_id}

