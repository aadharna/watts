import gym


class AlignedReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
        self.env = env
        _ = self.env.reset()
        self.env.enable_history(True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.win = None
        self.steps = -1
        self.play_length = None

    def step(self, action):
        assert(self.play_length is not None)
        action, reward, done, info = super().step(action)
        self.steps += 1
        if "PlayerResults" in info:
            self.win = info['PlayerResults']['1']
            # print(f"set win to: {self.win}")

        if self.win == 'Win':
            reward = 1 - (self.steps / self.play_length)
        elif self.win == 'Lose':
            reward = (self.steps / self.play_length) - 1
        else:
            reward = 0
        info['step'] = self.steps
        info['win']  = self.win
        info['r'] = reward

        return action, reward, done, info

    def reset(self, **kwargs):
        self.win = None
        self.steps = -1
        return self.env.reset(**kwargs)

    def __str__(self):
        return f"<aligned{str(self.env)}>"

def add_wrappers(str_list: list) -> list:
    wraps = []
    for w in str_list:
        if "Aligned" in w:
            wraps.append(AlignedReward)
        # add additional wrappers here
        # elif ...
        else:
            raise ValueError("Requested wrapper does not exist. Please make it.")

    return wraps

if __name__ == "__main__":
    import gym
    import griddly

    env = gym.make('GDY-Zelda-v0')
    print(str(env))
    env = AlignedReward(env)
    print(str(env))
    env.play_length = 500

    done = False
    rs = []
    infos = []
    while not done:
        ns, r, done, i = env.step(env.action_space.sample())
        rs.append(r)
        infos.append(i)

    print(rs)
    print(infos)
