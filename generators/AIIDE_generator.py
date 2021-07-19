import os
from typing import Tuple
import numpy as np
from copy import deepcopy
from itertools import product
from generators.base import BaseGenerator


class EvolutionaryGenerator(BaseGenerator):
    id = 0

    def __init__(self, level_string, file_args):
        """generator for maps that operates on tiles in a direct encoding

        :param level_string: string of level wwwwwwwwww\nw...
        :param file_args: arguments loaded from file via load_from_yaml and in the registry
        """
        super().__init__()

        self.args = file_args
        self.floor = self.args.floor[0]
        self.mechanics = self.args.mechanics

        if level_string[-1] != "\n":
            level_string += "\n"
        f = level_string.split('\n')[:-1]  # remove blank line.
        height = len(f)
        tile = [list(row) for row in f]

        npa = np.array(tile, dtype=str).reshape((height, -1))  # make into numpy array 9x13
        self.lvl_shape = npa.shape

        self._length = self.shape[0]
        self._height = self.shape[1]

        # set boundary values to 'w' in Zelda/GVGAI games
        range_height = list(product([0, self._length - 1], range(self._height)))
        range_length = list(product(range(self._length), [0, self._height - 1]))

        self.BOUNDARY = {'w': range_height + range_length}

        # sets the self.locations argument here because this function needs to a setter for outside methods
        self.update(level_string)
        self.string = str(self)

        self.id = EvolutionaryGenerator.id
        EvolutionaryGenerator.id += 1

    @property
    def shape(self):
        return self.lvl_shape

    def update(self, level):
        """
        Update Generator from flat lvl string
        :param level: flat lvl string with \n chars
        :return:
        """
        split_lvl = level.split('\n')[:-1]  # remove empty '' at the end

        o = np.array([['0'] * self._height] * self._length, dtype=str)
        for i in range(self._length):
            for j in range(self._height):
                o[i][j] = split_lvl[i][j]
        self.locations = self._parse_tile_world(o)

    def _parse_tile_world(self, tile_world):
        locations = {}
        # comb through world, extract positions for each element currently in world
        for i in range(len(tile_world)):
            for j in range(len(tile_world[i])):
                c = tile_world[i][j]
                if c not in locations:
                    locations[c] = []
                    locations[c].append((i, j))
                else:
                    locations[c].append((i, j))
        # add in user-specified sprites as empty lists.
        for char in self.mechanics:
            if char not in locations:
                locations[char] = []

        # separate out mutable walls vs non-mutable walls
        mutable_walls = list(set(locations['w']) - set(self.BOUNDARY['w']))
        locations['w'] = mutable_walls

        return locations

    def tile_world(self, locations):
        # numpy array
        npa = np.array([['0'] * self._height] * self._length, dtype=str)
        for k in locations.keys():
            for pos in locations[k]:
                npa[pos[0]][pos[1]] = k
        for k in self.BOUNDARY.keys():
            for pos in self.BOUNDARY[k]:
                npa[pos[0]][pos[1]] = k

        return npa

    def mutate(self, mutation_rate: float, **kwargs):  # -> EvolutionaryGenerator
        """randomly edit parts of the level!
        :param mutation_rate: e.g. 0.2
        :return: dict of location data for the entire level
        """
        locations = deepcopy(self.locations)

        def find_place_for_sprite(sprite):
            """find an empty location in the matrix for the sprite that is empty.

            :return: new (x, y) location
            """
            conflicting = True
            new_location = (0, 0)
            while conflicting:
                new_location = (np.random.randint(1, self._length),  # [, )  in, ex
                                np.random.randint(1, self._height))
                # print(f"potential location {new_location}")

                # don't overwrite Agent, goal, or key
                if not (new_location in [pos for k in self.args.singletons for pos in locations[k]] or
                        new_location in [pos for k in self.args.at_least_one for pos in locations[k] if
                                         len(locations[k]) == 1] or
                        new_location in [pos for k in self.BOUNDARY.keys() for pos in self.BOUNDARY[k]] or
                        new_location in locations[sprite]):
                    conflicting = False

            return new_location

        # if we manage to mutate:
        if np.random.rand() < mutation_rate:
            choices = np.arange(1, 4)

            ###
            # choices = [3, 3, 3] # TEMPORARY FOR THE EXPERIMENT OF CONSISTENT SHIFTING OF KEY AND DOORS.
            ###
            go_again = 0
            while go_again < 0.5:
                go_again = np.random.rand()

                mutationType = np.random.choice(choices, p=self.args.probs)  # [, )  in, ex

                # print(mutationType)
                # 1 -- remove sprite from scene               .... 20% chance
                # 2 -- spawn new sprite into the scene        .... 40% chance
                # 3 -- change location of sprite in the scene .... 40% chance
                if mutationType == 1:
                    skip = False
                    somethingToRemove = False
                    # choose a random sprite that has multiple instances of itself to remove
                    while not somethingToRemove:
                        sprite = np.random.choice(self.mechanics)
                        # print(f"removing {sprite}?")
                        # do not remove agent, cannot remove floor
                        if sprite in self.args.immortal:
                            # pick a new mutation
                            mutationType = np.random.choice(choices, p=[0, 0.5, 0.5])
                            # print(f"new mutation {mutationType}")
                            skip = True
                            break

                        # do not remove goal or key if there are only one of them
                        #  do not attempt to remove sprite when there are none of that type
                        elif len(locations[sprite]) <= 1:
                            if sprite in self.args.at_least_one or len(locations[sprite]) == 0:
                                mutationType = np.random.choice(choices, p=[0, 0.5, 0.5])
                                # print(f"new mutation {mutationType}")
                                skip = True
                                break
                        # else we have found something meaningful we can remove
                        else:
                            # print(f"removing {sprite}")
                            somethingToRemove = True
                    # choose location index in list of chosen sprite
                    if not skip:
                        ind = np.random.choice(len(locations[sprite]))
                        v = deepcopy(locations[sprite][ind])
                        # print(f"removed {v}")
                        locations[self.floor].append(v)
                        locations[sprite].pop(ind)

                # spawn a new sprite into the scene
                if mutationType == 2:
                    # choose a random sprite
                    spawned = False
                    while not spawned:
                        sprite = np.random.choice(self.mechanics)
                        if sprite in self.args.singletons:
                            continue
                        spawned = True
                    # print(f"spawning {sprite}?")
                    seed = np.random.choice(list(self.mechanics))
                    if len(locations[seed]) == 0:
                        pos = None
                    else:
                        ind = np.random.choice(len(locations[seed]))
                        pos = locations[seed][ind]

                    new_location = find_place_for_sprite(sprite=sprite)

                    # remove from whoever already has that new_location
                    for k in locations.keys():
                        if new_location in locations[k]:
                            rm_ind = locations[k].index(new_location)
                            locations[k].pop(rm_ind)
                            break

                    # add new sprite to the level
                    locations[sprite].append(new_location)

                # move an existing sprite in the scene
                if mutationType == 3:
                    moved = False
                    while not moved:
                        # choose a random viable sprite
                        sprite = np.random.choice(self.mechanics)
                        if len(list(locations[sprite])) == 0 or sprite in self.args.floor:
                            continue
                        moved = True

                    # choose location index in list of chosen sprite
                    ind = np.random.choice(len(locations[sprite]))
                    # where the sprite was previously
                    old = locations[sprite][ind]
                    # new location for sprite
                    new_location = find_place_for_sprite(sprite=sprite)
                    # print(f'moving {sprite} from {old} to {new_location}')

                    # remove whoever already has that new_location
                    # e.g. wall, floor
                    for k in locations.keys():
                        if new_location in locations[k]:
                            rm_ind = locations[k].index(new_location)
                            locations[k].pop(rm_ind)
                            break

                    locations[sprite].pop(ind)  # remove old position
                    # move sprite to new location
                    locations[sprite].append(new_location)
                    # fill previous spot with blank floor.
                    locations[self.floor].append(old)

        # remove anything that was in the boundary wall's spot.
        for k in locations.keys():
            for i, p in enumerate(locations[k]):
                if p in [pos for key in self.BOUNDARY.keys() for pos in self.BOUNDARY[key]]:
                    locations[k].pop(i)

        return EvolutionaryGenerator(self._to_str(locations), self.args)

    def generate_fn_wrapper(self):
        def _generate() -> Tuple[str, dict]:
            return str(self), {}
        return _generate

    def _to_str(self, location):
        stringrep = ""
        tile_world = self.tile_world(location)
        for i in range(len(tile_world)):
            for j in range(len(tile_world[i])):
                stringrep += tile_world[i][j]
                if j == (len(tile_world[i]) - 1):
                    stringrep += '\n'
        return stringrep

    def __str__(self):
        return self._to_str(self.locations)


if __name__ == "__main__":
    import os
    from utils.register import Registrar
    from utils.loader import load_from_yaml
    os.chdir('..')

    args = load_from_yaml('args.yaml')
    registry = Registrar(file_args=args)
    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = EvolutionaryGenerator(level_string=level_string, file_args=registry.get_generator_config)
    g2 = generator.mutate(0.88)

    print(str(generator))
    print(g2.generate_fn_wrapper()())
