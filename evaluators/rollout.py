from models.categorical_action_sampler import ActionSampler
import torch


def rollout(actor, env, device):
    """

    :param actor: NN solver to be evaluated
    :param env: An RLlibEnv to evaluate the solver in
    :param device: string of where you want the rollout to happen (e.g. cpu or gpu:0)
    :return: evaluation result state
    """
    sampler = ActionSampler(env.action_space)
    state = env.reset()
    # print(state.shape)
    done = False

    # use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device(device)
    actor.to(device)

    rewards = []
    states = []
    actions = []
    logps = []
    entropies = []
    win = False

    while not done:
        state = torch.FloatTensor([state]).to(device)
        logits, _ = actor({'obs': state}, None, None)
        torch_action, logp, entropy = sampler.sample(logits)
        action = torch_action.cpu().numpy()[0]
        next_state, reward, done, info = env.step(action)
        # env.render(observer='global')

        states.append(state)
        actions.append(torch_action)
        rewards.append(reward)
        logps.append(logp)
        entropies.append(entropy)
        state = next_state

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    return info, states, actions, rewards, win, logps, entropies
