from typing import Tuple

import numpy as np

from .base import BaseGenerator


class StaticGenerator(BaseGenerator):
    id = 0

    def __init__(self, level_string):
        """Static generator. This is initialized with a static level and that's all it will ever have.
        But this satisfies the definition of being a generator

        :param level_string: level to initialize the generator
        """
        BaseGenerator.__init__(self)
        self.level = level_string
        if level_string[-1] != "\n":
            level_string += "\n"
        tile = [row.split() for row in level_string.split('\n')[:-1]]  # remove blank line.
        height = len(tile)

        npa = np.array(tile, dtype=str).reshape((height, -1))  # make into numpy array 9x13
        self.lvl_shape = npa.shape
        self.id = StaticGenerator.id
        StaticGenerator.id += 1

    @property
    def shape(self):
        return self.lvl_shape

    def mutate(self, **kwargs):
        return StaticGenerator(self.level)

    def update(self, level):
        self.level = level

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            return self.level, {}
        return _generate

    def __str__(self):
        return self.level


if __name__ == "__main__":
    level_string = '''w w w w w w w w w w w w w\nw . . . . + e . . . . . w\nw . . . . . . . . . . . w\nw . . A . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . . . . . . . w\nw . . . . . w . . . . . w\nw . g . . . . . . . . . w\nw w w w w w w w w w w w w\n'''
    generator = StaticGenerator(level_string)
    print(str(generator))
    print(generator.generate_fn_wrapper()())
