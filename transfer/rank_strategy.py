from typing import Dict, Any

import ray
import numpy as np
from itertools import product


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

        # n_gens = len(generators)
        # n_solvs = len(solvers)
        results = ray.get(combo_refs)
        # returned_scores = []
        # returned_generators = []
        # returned_solvers = []
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
            # todo track this info for analysis purposes
            # print(f"updated {generator_id} to {best_id}")

        #         returned_scores.append(sum(r['score']))
        #         returned_generators.append(r['generator_id'])
        #         returned_solvers.append(r['solver_id'])
        #
        # scores = np.array(returned_scores).reshape(n_gens, n_solvs).T  # this will be square for POET
        # scores[2][2] = 100
        # gens = np.array(returned_generators).reshape(n_gens, n_solvs)
        # solvs = np.array(returned_solvers).reshape(n_gens, n_solvs)
        # best_result_index_per_gen = np.argmax(scores, axis=1)
        #
        #
        # for i, (best_pair_id, generator_id) in enumerate(zip(best_result_index_per_gen, id_map)):
        #     # todo change these indicies into the pair.id to track who is leaving where and going to where
        #
        #     # if id_map @ best_solver_index == generator_id
        #     #   then we would just be replacing ourself, so there is no need to track a transfer
        #     best_solver = id_map.index(solvs[i][best_pair_id])
        #     new_weights[generator_id] = solvers[best_solver].state_dict()

        return new_weights
