from typing import Dict, Any
from itertools import product

import numpy as np
import ray


class RankStrategy:
    def transfer(self, solver_list: list, generator_list: list, id_map: list) -> Dict[int, Any]:
        raise NotImplementedError


class GetBestSolver(RankStrategy):
    def __init__(self, scorer):
        self.scorer = scorer

    def transfer(self, solver_list: list, generator_list: list, id_map: list) -> Dict[int, Any]:
        """Run the transfer tournament; take in a solver list and a generator list.

        :param solver_list:
        :param generator_list:
        :param id_map:
        :return: dict of new weights indexed by a pair_id.
        """

        solvers, solver_idxs = zip(*solver_list)
        generators, generator_idxs = zip(*generator_list)
        solver_generator_combo_id = list(product(id_map, id_map))
        combo_refs = self.scorer.score(solvers=solvers,
                                       generators=generators,
                                       id_map=solver_generator_combo_id)

        results = ray.get(combo_refs)
        new_weights = {}
        for i, generator_id in enumerate(id_map):
            best_s = -np.inf
            best_w = ray.get(solvers[i].get_weights.remote())
            best_id = generator_id
            for j, r in enumerate(results):
                if generator_id == r['generator_id']:
                    score = r[0]['score']
                    solver = r['solver_id']
                    if score > best_s:
                        best_s = score
                        best_w = ray.get(solvers[id_map.index(solver)].get_weights.remote())
                        best_id = solver
            new_weights[generator_id] = (best_w, best_id)
        return new_weights
