from models.categorical_action_sampler import ActionSampler
import torch


def rollout(actor, env):
    """

    :param actor: NN solver to be evaluated
    :param env: An RLlibEnv to evaluate the solver in
    :return: evaluation result state
    """
    sampler = ActionSampler(env.action_space)
    state = env.reset()
    # print(state.shape)
    done = False

    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    actor.to(device)

    rewards = []
    states = []
    actions = []
    win = False

    while not done:
        state = torch.FloatTensor([state]).to(device)
        x, _ = actor({'obs': state}, None, None)
        action, logp, entropy = sampler.sample(x)
        action = action.cpu().numpy()[0]
        next_state, reward, done, info = env.step(action)
        # env.render(observer='global')

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    return info, states, actions, rewards, win
