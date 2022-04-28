import ray
import numpy as np
from itertools import product


class ScoreStrategy:
    def score(self, solvers, generators, id_map) -> dict:
        raise NotImplementedError


class ZeroShotCartesian(ScoreStrategy):
    def __init__(self, config):
        self.env_config = config

    def score(self, solvers, generators, id_map) -> np.ndarray:
        """run the evaluate function on the cartesian product of solvers and generators

        @param solvers: list of Solver objects
        @param generators: list of callable string levels
        @param id_map: list of tuples of (solver_id, generator_id) also created with product (id, id)
        :return: an n x m x 3 tensor. Where n = len(solvers), m = len(generators), 3 = len((gen_id, sol_id, score))
        """

        refs = []
        for i, (s, g) in enumerate(product(solvers, generators)):
            solver_id, generator_id = id_map[i]
            self.env_config['level_string'], _ = g.generate_fn_wrapper()()
            refs.append(s.evaluate.remote(env_config=self.env_config,
                                          solver_id=solver_id,
                                          generator_id=generator_id))
        results = ray.get(refs)

        id_map = np.array(id_map)
        solver_ids = set(id_map[:, 0])
        generator_ids = set(id_map[:, 1])

        tournament_results = {i: [] for i in solver_ids}

        for i, generator_id in enumerate(generator_ids):
            for j, r in enumerate(results):
                if generator_id == r['generator_id']:
                    score = r[0]['score']
                    solver_id = r['solver_id']
                    tournament_results[solver_id].append((generator_id, solver_id, score))

        tt = []
        for k, v in tournament_results.items():
            tt.append(v)

        tt = np.array(tt).squeeze()  # we squeeze because that'll clean up the [n,1,3] case into [n,3] but
        #   the [n,n,3] case will stay [n,n,3].

        return tt
