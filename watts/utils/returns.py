

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * masks[i]
        returns.insert(0, R)
    return returns


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns