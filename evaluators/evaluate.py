import os
import ray
import gym
import griddly
from griddly import gd

from levels.zelda_action_interpreter import interpretAction

import torch


@ray.remote
def evaluate_agent_on_level(gym_factory_monad, network_factory_monad, level_string, actor_critic_weights):
    """

    :param gym_factory_monad: Factory_make function for gym.Env
    :param network_factory_monad: Factory_make function for NN
    :param level_string: what level do you want to load into the game defined by the above gdy file
    :param actor_critic_weights: weights for your actor-critic network
    :return:
    """

    env = gym_factory_monad()  # use built-in env_config dict
    actor = network_factory_monad()

    actor.load_state_dict(actor_critic_weights)
    state = env.reset(level_string=level_string)
    # print(state.shape)
    done = False

    use_cuda = torch.cuda.is_available()
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

    return {'score': rewards, 'win': win == "Win", 'info': info}


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

    gameFactory = GridGameFactory(args, [AlignedReward], registrar=registrar)
    networkFactory = NetworkFactory(registrar=registrar)

    network = networkFactory.make()()
    actor_critic_weights = network.state_dict()

    rewards_future = evaluate_agent_on_level.remote(gym_factory_monad=gameFactory.make(),
                                                    network_factory_monad=networkFactory.make(),
                                                    level_string=level_string,
                                                    actor_critic_weights=actor_critic_weights)

    # rewards, returns, log_probs, values, states, actions, advantage, entropy = bar

    foo = ray.get(rewards_future)
    print(foo)
    print(sum(foo['score']))
    ray.shutdown()
