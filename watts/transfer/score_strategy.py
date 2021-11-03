import ray
from itertools import product


class ScoreStrategy:
    def score(self, solvers, generators, id_map) -> list:
        raise NotImplementedError


class ZeroShotCartesian(ScoreStrategy):
    def __init__(self, config):
        self.env_config = config

    def score(self, solvers, generators, id_map) -> list:
        """run the evaluate function on the cartesian product of solvers and generators

        :param solvers: list of Solver objects
        :param generators: list of callable string levels
        :param id_map: list of tuples of (solver_id, generator_id) also created with product (id, id)
        :return: list of refs (and list of ids) for the evaluated solver-generator pairs
        """

        refs = []
        for i, (s, g) in enumerate(product(solvers, generators)):
            solver_id, generator_id = id_map[i]
            self.env_config['level_string'], _ = g.generate_fn_wrapper()()
            refs.append(s.evaluate_buffer.remote(env_config=self.env_config,
                                                 solver_id=solver_id,
                                                 generator_id=generator_id))
        return ray.get(refs)
