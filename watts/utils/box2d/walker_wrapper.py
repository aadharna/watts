import gym
from watts.utils.box2d.biped_walker_custom import Env_config


def stringToNamedTuple(asdisctString):
    dict_form = eval(asdisctString)
    return Env_config(**dict_form)


class OverrideWalker(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
        self.step_counter = 0
        self.track = False

    def reset(self, level_string=""):
        self.step_counter = 0
        if bool(level_string):
            new_config = stringToNamedTuple(level_string)
            self.env.set_env_config(new_config)
        return self.env.reset()

    def step(self, action):
        ns, r, d, i = self.env.step(action)
        if 'finish' in i:
            i['PlayerResults'] = {}
            i['PlayerResults']['1'] = 'Win' if i['finish'] else 'Lose'
        if self.track:
            i['position'] = self.env.hull.position
            i['velocity'] = self.env.hull.linearVelocity
            i['joints']   = self.env.joints
            i['step']     = self.step_counter
        self.step_counter += 1
        return ns, r, d, i

    # here for compatability with griddly::rllibenv
    def enable_history(self, enable=True):
        self.track = enable

    # here for compatability with griddly::rllibenv
    def on_episode_start(self, worker_idx, env_idx):
        pass

    # here for compatability with griddly::rllibenv
    def render(self, mode="human", **kwargs):
        observer = kwargs.get('observer', None)
        # replace 'global' with e.g. 'human' render mode
        if observer is not None:
            observer = mode
            del kwargs['observer']
        self.env.render(mode=observer, **kwargs)

    # here for compatability with griddly::rllibenv
    #  specifically, used when releasing resources from the Solver class back to the OS
    @property
    def game(self):
        class Foo:
            def release(self):
                return
        return Foo()
