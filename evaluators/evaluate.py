import os
import ray
import gym
import griddly
from griddly import gd

from levels.zelda_action_interpreter import interpretAction

import torch


@ray.remote
def evaluate_agent_on_level(gym_factory_monad,
                            rllib_env_config,
                            level_string_monad,
                            network_factory_monad,
                            actor_critic_weights,
                            **kwargs):
    """

    :param gym_factory_monad: Factory_make function for env
    :param rllib_env_config: dictionary of necessary information to make the env
    :param level_string_monad: what level do you want to load into the game defined by the above gdy file
    :param network_factory_monad: Factory_make function for NN
    :param actor_critic_weights: weights for your actor-critic network
    :return:
    """

    env = gym_factory_monad(rllib_env_config)
    actor = network_factory_monad()

    actor.load_state_dict(actor_critic_weights)
    # todo will probably have to change this to first instantiate a generator model
    # and then query it for the levels.
    #  That will allow something like PAIRED to function.
    state = env.reset(level_string=level_string_monad())
    # print(state.shape)
    done = False

    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    actor.to(device)

    rewards = []
    win = False

    while not done:
        state = torch.FloatTensor([state]).to(device)
        x, _ = actor({'obs': state}, None, None)
        _, torch_action = torch.max(x.squeeze(), 0)
        action = interpretAction(torch_action.cpu().numpy())
        next_state, reward, done, info = env.step(action)
        # env.render(observer='global')

        rewards.append(reward)
        state = next_state

    env.close()
    env.game.release()

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    return {'score': rewards, 'win': win == "Win", 'info': info,
            'solver_id': kwargs.get('solver_id', 0), 'gen_id': kwargs.get('gen_id', 0)}


if __name__ == "__main__":
    from utils.register import Registrar
    from utils.loader import load_from_yaml
    from utils.gym_wrappers import AlignedReward

    from gym_factory import GridGameFactory
    from network_factory import NetworkFactory

    os.chdir('..')

    ray.init()

    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''

    args = load_from_yaml(os.path.join('args.yaml'))

    registrar = Registrar(file_args=args)

    gameFactory = GridGameFactory(registrar.env_name,
                                  env_wrappers=[AlignedReward])

    networkFactory = NetworkFactory(registrar.network_name, registrar.get_nn_build_info)

    network = networkFactory.make()()
    actor_critic_weights = network.state_dict()

    rewards_future = evaluate_agent_on_level.remote(gym_factory_monad=gameFactory.make(),
                                                    rllib_env_config=registrar.get_config_to_build_rllib_env,
                                                    network_factory_monad=networkFactory.make(),
                                                    level_string_monad=lambda: level_string,
                                                    actor_critic_weights=actor_critic_weights,
                                                    solver_id=90,
                                                    gen_id=0)

    # rewards, returns, log_probs, values, states, actions, advantage, entropy = bar

    foo = ray.get(rewards_future)
    print(foo)
    print(sum(foo['score']))
    ray.shutdown()
