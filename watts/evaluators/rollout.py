import ray
import torch
from collections import namedtuple

from watts.models.action_sampler import ActionSampler

Rollout_results = namedtuple('Rollout_results',
                             ['info', 'states', 'values', 'actions', 'rewards', 'win', 'logps', 'entropies', 'dones'])


def rollout(actor, env, device) -> Rollout_results:
    """

    :param actor: NN solver to be evaluated
    :param env: An RLlibEnv to evaluate the solver in
    :param device: string of where you want the rollout to happen (e.g. cpu or gpu:0)
    :return: evaluation result state
    """
    sampler = ActionSampler(env.action_space)
    state = env.reset()
    done = False

    device = torch.device(device)
    actor.to(device)

    rewards = []
    values = []
    states = []
    actions = []
    logps = []
    entropies = []
    dones = []
    win = False

    while not done:
        state = torch.FloatTensor([state]).to(device)
        logits, _ = actor({'obs': state}, None, None)
        value = actor.value_function()
        torch_action, logp, entropy = sampler.sample(logits)
        action = torch_action.cpu().numpy()[0]
        next_state, reward, done, info = env.step(action)
        # env.render(observer='global')

        states.append(state)
        actions.append(torch_action)
        values.append(value)
        rewards.append(reward)
        logps.append(logp)
        entropies.append(entropy)
        dones.append(not done)
        state = next_state

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    results = Rollout_results(info=info,
                              states=states,
                              values=values,
                              actions=actions,
                              rewards=rewards,
                              win=win,
                              logps=logps,
                              entropies=entropies,
                              dones=dones)
    return results


@ray.remote
def remote_rollout(nn_make_fn, env_make_fn, nn_weights, env_config):
    agent = nn_make_fn(nn_weights)
    env = env_make_fn(env_config)
    return rollout(agent, env, 'cpu')


if __name__ == '__main__':
    import os
    import ray
    from watts.utils.loader import load_from_yaml
    from watts.utils.register import Registrar
    from watts.utils.gym_wrappers import add_wrappers
    from watts.gym_factory import WalkerFactory
    from watts.network_factory import NetworkFactory
    while 'poet_distributed.py' not in os.listdir('.'):
        os.chdir('..')
    print(os.listdir('.'))

    ray.init()

    args = load_from_yaml(fpath=os.path.join('sample_args', 'walker_args.yaml'))
    args.exp_name = 'poet'

    registry = Registrar(file_args=args)
    # game_schema = GameSchema(registry.gdy_file) # Used for GraphValidator
    wrappers = add_wrappers(args.wrappers)
    # gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    gym_factory = WalkerFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info)

    env = gym_factory.make()(registry.get_config_to_build_rllib_env)
    nn = network_factory.make()({})
    result = rollout(nn, env, 'cpu')
    print(result.rewards)
    env.viewer.close()

    ray.shutdown()