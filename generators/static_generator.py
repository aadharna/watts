
from generators.base import BaseGenerator

class StaticGenerator(BaseGenerator):

    def __init__(self, level_string):
        """Static generator. This is initialized with a static level and that's all it will ever have.
        But this satisfies the definition of being a generator

        :param level_string: level to initialize the generator
        """
        BaseGenerator.__init__(self)
        self.level = level_string

    def mutate(self, **kwargs):
        pass

    def update_from_lvl_string(self, level_string):
        self.level = level_string

    def generate(self):
        def _generate():
            return self.level
        return _generate

    def __str__(self):
        return self.level

if __name__ == "__main__":
    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = StaticGenerator(level_string)
    print(str(generator))
    print(generator.generate()())
