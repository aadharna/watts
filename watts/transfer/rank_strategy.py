from typing import Dict, Any
from itertools import product

import numpy as np
import ray


class RankStrategy:
    def transfer(self, solver_list: list, generator_list: list) -> Dict[int, Any]:
        raise NotImplementedError


class Noop(RankStrategy):
    def transfer(self, solver_list: list, generator_list: list) -> Dict[int, Any]:
        return {}


class GetBestSolver(RankStrategy):
    def __init__(self, scorer):
        self.scorer = scorer
        self.tournaments = {}
        self.t = 0

    def transfer(self, solver_list: list, generator_list: list) -> Dict[int, Any]:
        """Run the transfer tournament; take in a solver list and a generator list.

        :param solver_list:
        :param generator_list:
        :param id_map:
        :return: dict of new weights indexed by a pair_id.
        """
        self.t += 1

        solvers, solver_idxs = zip(*solver_list)
        generators, generator_idxs = zip(*generator_list)
        solver_generator_combo_id = list(product(solver_idxs, generator_idxs))
        results = self.scorer.score(solvers=solvers,
                                    generators=generators,
                                    id_map=solver_generator_combo_id)

        tournaments = {i: [] for i in solver_idxs}

        for i, generator_id in enumerate(generator_idxs):
            for j, r in enumerate(results):
                if generator_id == r['generator_id']:
                    score = r[0]['score']
                    solver = r['solver_id']
                    tournaments[solver].append((generator_id, solver, score))

        tt = []
        for k, v in tournaments.items():
            tt.append(v)

        tt = np.array(tt)
        self.tournaments[self.t] = tt
        generator_id_matrix = tt[:, :, 0]
        solver_id_matrix = tt[:, :, 1]
        # slice the tensor into a matrix (remove the index chasing matrix)
        # argmax the matrix across the agents for each task.
        best_indicies = np.argmax(tt[:, :, 2], axis=0)


        """
        Result Matrix (aka tt[:, :, 2]):
                    T0      T1      T2      T3      T4
        array([A0  [-0.902,  0.   , -0.212,  0.   ,  0.   ],
               A1  [ 0.264,  0.   , -0.846,  0.   ,  0.88 ],
               A2  [-0.982, -0.852,  0.   , -0.908, -0.792],
               A3  [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
               A4  [ 0.   ,  0.   ,  0.152,  0.   ,  0.896]]
        
        Ans: array([1, 0, 4, 0, 4], dtype=int64)
        """
        new_weights = {}
        for i in range(generator_id_matrix.shape[0]):
            best_solver_id = solver_id_matrix.T[i, best_indicies[i]]
            for j, _id in enumerate(solver_idxs):
                if _id == best_solver_id:
                    new_weights[generator_idxs[i]] = (
                        ray.get(solvers[j].get_weights.remote()),
                        best_solver_id
                    )

        return new_weights
