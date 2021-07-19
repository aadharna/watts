from typing import Tuple

from generators.base import BaseGenerator
import numpy as np


class RandomSelectionGenerator(BaseGenerator):

    def __init__(self, level_strings):
        """Static generator. This is initialized with a static level and that's all it will ever have.
        But this satisfies the definition of being a generator

        :param level_string: level to initialize the generator
        """
        BaseGenerator.__init__(self)
        self.levels = level_strings
        self.current_level = 0
        level_string = level_strings[0]
        if level_string[-1] != "\n":
            level_string += "\n"
        f = level_string.split('\n')[:-1]  # remove blank line.
        height = len(f)
        tile = [list(row) for row in f]

        npa = np.array(tile, dtype=str).reshape((height, -1))  # make into numpy array 9x13
        self.lvl_shape = npa.shape

    @property
    def shape(self):
        return self.lvl_shape

    def mutate(self, **kwargs):
        return RandomSelectionGenerator(self.levels)

    def update(self, level):
        self.levels = level

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            self.current_level = np.random.randint(0, len(self.levels))
            return self.levels[self.current_level], {'id': self.current_level}
        return _generate

    def __str__(self):
        return self.levels[self.current_level]


if __name__ == "__main__":
    lvls = [
        '''wwwwwwwwwwwww\nw....+.+++..w\nw....www....w\nw..A........w\nw...........w\nw...........w\nwwwwwww.....w\nw.g.......++w\nwwwwwwwwwwwww\n''',
        '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n''',
        '''wwwwwwwwwwwww\nw....+......w\nw........wwww\nw..A........w\nw...........w\nw.....wwwwwww\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n''',
        '''wwwwwwwwwwwww\nwwww.+......w\nw...........w\nw..A......eew\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    ]

    generator = RandomSelectionGenerator(lvls)
    print(str(generator))
    new_lvl = generator.generate_fn_wrapper()()
    print(str(generator))
