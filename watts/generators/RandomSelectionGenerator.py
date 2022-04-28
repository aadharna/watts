from typing import Tuple

import numpy as np

from watts.generators.base import BaseGenerator


class RandomSelectionGenerator(BaseGenerator):

    def __init__(self, level_strings):
        """Random selection generator. This is initialized with a list of static levels and that's all it will ever have.
        The generator can sample one of these static levels on demand.
        This simple unchanging generator satisfies the definition of being a generator

        :param level_string: level to initialize the generator
        """
        BaseGenerator.__init__(self)
        self.levels = level_strings
        self.current_level = 0
        level_string = level_strings[0]
        if level_string[-1] != "\n":
            level_string += "\n"
        tile = [row.split() for row in level_string.split('\n')[:-1]]  # remove blank line.
        height = len(tile)

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
        '''w w w w w w w w w w w w w\nw . . . . + . + + + . . w\nw . . . . w w w . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw w w w w w w . . . . . w\nw . g . . . . . . . + + w\nw w w w w w w w w w w w w\n''',
        '''w w w w w w w w w w w w w\nw . . . . + e . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n''',
        '''w w w w w w w w w w w w w\nw . . . . + . . . . . . w\nw . . . . . . . . w w w w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w w w w w w w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n''',
        '''w w w w w w w w w w w w w\nw w w w . + . . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . e e w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n'''
    ]

    generator = RandomSelectionGenerator(lvls)
    print(str(generator))
    new_lvl = generator.generate_fn_wrapper()()
    print(str(generator))
