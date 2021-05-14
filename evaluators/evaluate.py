from levels.zelda_action_interpreter import interpret_action

import torch


def evaluate(actor, env):
    """

    :param actor: actor being evaluated (ex: NN)
    :param env: env to evaluate in
    :return: evaluation result state
    """

    state = env.reset()
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
        action = interpret_action(torch_action.cpu().numpy())
        next_state, reward, done, info = env.step(action)
        # env.render(observer='global')

        rewards.append(reward)
        state = next_state

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    return info, rewards, win
