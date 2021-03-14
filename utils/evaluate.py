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

    env = gym_factory_monad()
    actor = network_factory_monad()

    actor.load_state_dict(actor_critic_weights)
    state = env.reset(level_string=level_string)
    print(state.shape)
    done = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    actor.to(device)

    rewards = []

    while not done:
        state = torch.FloatTensor([state]).to(device)
        x, _ = actor({'obs': state}, None, None)
        _, torch_action = torch.max(x.squeeze(), 0)
        action = interpretAction(torch_action.cpu().numpy())
        next_state, reward, done, _ = env.step(action)
        # env.render(observer='global')

        rewards.append(reward)
        state = next_state
    env.close()

    return rewards


if __name__ == "__main__":
    from utils.register import register_env_with_rllib
    from utils.loader import load_from_yaml
    from utils.gym_wrappers import AlignedReward

    from gym_factory import GridGameFactory
    from network_factory import NetworkFactory

    os.chdir('..')

    ray.init()

    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''

    args = load_from_yaml(os.path.join('args.yaml'))

    name, nActions, actSpace, obsSpace, observer = register_env_with_rllib(file_args=args)

    gameFactory = GridGameFactory(args, name, nActions, actSpace, obsSpace, observer, [AlignedReward])
    networkFactory = NetworkFactory(obs_space=obsSpace, action_space=actSpace,
                         num_outputs=nActions, model_config={}, name='AIIDE_PINSKY_MODEL')

    network = networkFactory.make()()
    actor_critic_weights = network.state_dict()

    rewards_future = evaluate_agent_on_level.remote(gym_factory_monad=gameFactory.make(),
                                                    network_factory_monad=networkFactory.make(),
                                                    level_string=level_string,
                                                    actor_critic_weights=actor_critic_weights)

    # rewards, returns, log_probs, values, states, actions, advantage, entropy = bar

    foo = ray.get(rewards_future)
    print(sum(foo))
    ray.shutdown()
