import copy
import time
import uuid
import numpy as np
from typing import Tuple, Dict

from watts.generators.base import BaseGenerator
from watts.utils.box2d.biped_walker_custom import Env_config


def name_env_config(ground_roughness,
                    pit_gap,
                    stump_width, stump_height, stump_float,
                    stair_width, stair_height, stair_steps):
    env_name = 'r' + str(ground_roughness)
    if pit_gap:
        env_name += '.p' + str(pit_gap[0]) + '_' + str(pit_gap[1])
    if stump_width:
        env_name += '.b' + str(stump_width[0]) + '_' + str(stump_height[0]) + '_' + str(stump_height[1])
    if stair_steps:
        env_name += '.s' + str(stair_steps[0]) + '_' + str(stair_height[1])

    return env_name


class WalkerConfigGenerator(BaseGenerator):
    """This is a copy/paste and massaging of the Reproducer class
    from POET/Enhanced POET into Watts.
    https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/reproduce_ops.py
    """
    id = 0

    def __init__(self, parent_env_config, categories=('stump', 'pit', 'roughness', 'stair')):
        super().__init__()
        self.env_config = parent_env_config
        self.rs = np.random.RandomState(int(time.time()))
        # in poet, this argument reads:
        # self.categories = list(args.envs)
        #  args is the argparse args list
        #  The argparse Namespace object has envs as: parser.add_argument('--envs', nargs='+')
        #    however, if you look in the run_remote.sh script, this values is filled in as
        #      --envs stump pit roughness
        self.categories = categories

    def populate_array(self, arr, default_value,
                       interval=0, increment=0., enforce=False, max_value=[]):
        assert isinstance(arr, list)
        if len(arr) == 0 or enforce:
            arr = list(default_value)
        elif len(max_value) == 2:
            choices = []
            for change0 in [increment, 0.0, -increment]:
                arr0 = np.round(arr[0] + change0, 1)
                if arr0 > max_value[0] or arr0 < default_value[0]:
                    continue
                for change1 in [increment, 0.0, -increment]:
                    arr1 = np.round(arr[1] + change1, 1)
                    if arr1 > max_value[1] or arr1 < default_value[1]:
                        continue
                    if change0 == 0.0 and change1 == 0.0:
                        continue
                    if arr0 + interval > arr1:
                        continue

                    choices.append([arr0, arr1])

            num_choices = len(choices)
            if num_choices > 0:
                idx = self.rs.randint(num_choices)
                # print(choices)
                # print("we pick ", choices[idx])
                arr[0] = choices[idx][0]
                arr[1] = choices[idx][1]

        return arr

    def mutate(self, **kwargs):
        no_mutate = kwargs.get('no_mutate', False)
        parent = copy.deepcopy(self.env_config)

        ground_roughness = parent.ground_roughness
        pit_gap = list(parent.pit_gap)
        stump_width = list(parent.stump_width)
        stump_height = list(parent.stump_height)
        stump_float = list(parent.stump_float)
        stair_height = list(parent.stair_height)
        stair_width = list(parent.stair_width)
        stair_steps = list(parent.stair_steps)

        if no_mutate:
            child_name = str(uuid.uuid4())
        else:
            if 'roughness' in self.categories:
                ground_roughness = np.round(ground_roughness + self.rs.uniform(-0.6, 0.6), 1)
                max_roughness = 10.0
                if ground_roughness > max_roughness:
                    ground_roughness = max_roughness

                if ground_roughness <= 0.0:
                    ground_roughness = 0.0

            if 'pit' in self.categories:
                pit_gap = self.populate_array(pit_gap, [0, 0.8], increment=0.4, max_value=[8.0, 8.0])

            if 'stump' in self.categories:
                sub_category = '_h'
                enforce = (len(stump_width) == 0)

                if enforce or sub_category == '_w':
                    stump_width = self.populate_array(stump_width, [1, 2], enforce=enforce)

                if enforce or sub_category == '_h':
                    stump_height = self.populate_array(stump_height, [0, 0.4], increment=0.2, enforce=enforce,
                                                       max_value=[5.0, 5.0])

                stump_float = self.populate_array(stump_float, [0, 1], enforce=True)

            if 'stair' in self.categories:
                sub_category = '_h'  # self.rs.choice(['_s', '_h'])
                enforce = (len(stair_steps) == 0)

                if enforce or sub_category == '_s':
                    stair_steps = self.populate_array(stair_steps, [1, 2], interval=1, increment=1, enforce=enforce,
                                                      max_value=[9, 9])
                    stair_steps = [int(i) for i in stair_steps]

                if enforce or sub_category == '_h':
                    stair_height = self.populate_array(stair_height, [0, 0.4], increment=0.2, enforce=enforce,
                                                       max_value=[5.0, 5.0])

                stair_width = self.populate_array(stump_width, [4, 5], enforce=True)

            child_name = name_env_config(ground_roughness, pit_gap, stump_width, stump_height, stump_float, stair_width,
                                         stair_height, stair_steps)

        child = Env_config(name=child_name,
                           ground_roughness=ground_roughness,
                           pit_gap=pit_gap,
                           stump_width=stump_width,
                           stump_height=stump_height,
                           stump_float=stump_float,
                           stair_height=stair_height,
                           stair_width=stair_width,
                           stair_steps=stair_steps)

        return WalkerConfigGenerator(parent_env_config=child, categories=self.categories)

    # TODO: If we wanted to override the current level
    def update(self, level):
        if isinstance(level, Env_config):
            self.env_config = level
        elif isinstance(level, str):
            try:
                from watts.utils.box2d.walker_wrapper import stringToNamedTuple
                self.env_config = stringToNamedTuple(level)
            except NameError as e:
                raise ValueError('Passed in env_config string was corrupted.')
        else:
            raise ValueError("We passed in a fucking not-level. Passed in object is not an env_config")

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            return str(self), {}
        return _generate

    def __str__(self):
        return str(self.env_config._asdict())

    @property
    def shape(self):
        return (200,)


if __name__ == "__main__":
    from watts.utils.box2d.biped_walker_custom import Env_config, DEFAULT_ENV
    g = WalkerConfigGenerator(parent_env_config=DEFAULT_ENV)
    g2 = g.mutate()
    print(str(g))
    print(str(g2))

