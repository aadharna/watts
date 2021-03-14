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
        self.steps = 0
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

        return action, reward, done, info

    def __str__(self):
        return f"<aligned{str(self.env)}>"


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
