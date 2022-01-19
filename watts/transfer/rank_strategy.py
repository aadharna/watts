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
        tournament_results = self.scorer.score(solvers=solvers,
                                               generators=generators,
                                               id_map=solver_generator_combo_id)

        tt = []
        for k, v in tournament_results.items():
            tt.append(v)

        tt = np.array(tt).squeeze() # we squeeze because that'll clean up the [n,1,3] case into [n,3] but
                                    #   the [n,n,3] case will stay [n,n,3].
        self.tournaments[self.t] = tt
        # here we're using the generalized slicing of ... isntead of manually indicating the number of
        #  dimensions to go through and are selecting the various slices but keeping length x width
        generator_id_matrix = tt[..., 0]
        solver_id_matrix = tt[..., 1]
        # slice the tensor into a matrix (remove the index chasing matrix)
        # argmax the matrix across the agents for each task.
        best_indicies = np.argmax(tt[..., 2], axis=0)


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
        for i, g_id in enumerate(generator_idxs):
            if solver_id_matrix.ndim == 2:
                best_solver_id = solver_id_matrix.T[i, best_indicies[i]]
            elif solver_id_matrix.ndim == 1:
                best_solver_id = solver_id_matrix[best_indicies]
            elif solver_id_matrix.ndim == 0:
                best_solver_id = solver_id_matrix
            else:
                raise ValueError('result tensor is other than expected size; supported shapes are [n,n,3], [n,3], [3]')
            for j, s_id in enumerate(solver_idxs):
                if s_id == best_solver_id:
                    new_weights[g_id] = (
                        ray.get(solvers[j].get_weights.remote()),
                        best_solver_id
                    )

        return new_weights


class GetBestZeroOrOneShotSolver(RankStrategy):
    """This is the transfer strategy that POET uses in:
    https://arxiv.org/abs/1901.01753 see page 28

    """
    def __init__(self, scorer, default_trainer_config):
        self.internal_transfer_strategy = GetBestSolver(scorer=scorer)
        self.trainer_config = default_trainer_config
        self.direct_transfers = {}
        self.proposal_transfers = {}
        self.t = 0

    def transfer(self, solver_list: list, generator_list: list) -> Dict[int, Any]:
        self.t += 1

        solvers, solver_idxs = zip(*solver_list)
        generators, generator_idxs = zip(*generator_list)
        solver_generator_combo_id = list(product(solver_idxs, generator_idxs))

        # in POET's language these are (potential) direct transfers
        best_zero_shot_weights = self.internal_transfer_strategy.transfer(solver_list, generator_list)
        zero_shot_data = self.internal_transfer_strategy.tournaments[self.internal_transfer_strategy.t]

        # take one opt step for each Pair
        refs = []
        for i, (s, g) in enumerate(zip(solvers, generators)):
            # This does an in-place update of the weights
            refs.append(s.optimize.remote(trainer_config=self.trainer_config,
                                          level_string_monad=g.generate_fn_wrapper(),
                                          pair_id=solver_idxs[i]))
        _ = ray.get(refs)

        # in POET's language, these are (potential) proposal transfers
        best_one_shot_weights = self.internal_transfer_strategy.transfer(solver_list, generator_list)
        one_shot_data = self.internal_transfer_strategy.tournaments[self.internal_transfer_strategy.t]

        new_weights = {}
        proposal_transfers = 0
        direct_transfers = 0
        # split apart the tensors
        zero_generator_id_matrix = zero_shot_data[..., 0]
        zero_solver_id_matrix = zero_shot_data[..., 1]
        zero_best_indicies = np.argmax(zero_shot_data[..., 2], axis=0)
        one_generator_id_matrix = one_shot_data[..., 0]
        one_solver_id_matrix = one_shot_data[..., 1]
        one_best_indicies = np.argmax(one_shot_data[..., 2], axis=0)
        for i, g_id in enumerate(generator_idxs):
            w_0, s_id_0 = best_zero_shot_weights[g_id]
            w_1, s_id_1 = best_one_shot_weights[g_id]
            # POET does transfer for one env at a time, here we do the entire set once.
            # POET transfer for env k is argmax([theta_1, ..., theta_m, theta_1', ..., theta_m'])
            #    we have done:
            #         argmax(argmax([theta_1, ..., theta_m]), argmax([theta_1', ..., theta_m']))
            #    and the implemented version is equivalent to POET's version
            #    since max([1, 2, 3, 4, 5, 6, 7, 8]) = max(max([1, 2, 3, 4]), max([5, 6, 7, 8]))
            #
            if zero_shot_data.ndim == 3:
                if zero_shot_data[..., 2][i, zero_best_indicies[i]] > one_shot_data[..., 2][i, one_best_indicies[i]]:
                    new_weights[g_id] = (w_0, s_id_0)
                    direct_transfers += 1
                else:
                    new_weights[g_id] = (w_1, s_id_1)
                    proposal_transfers += 1
            elif zero_shot_data.ndim == 2:
                if zero_shot_data[..., 2][zero_best_indicies] > one_shot_data[..., 2][one_best_indicies]:
                    new_weights[g_id] = (w_0, s_id_0)
                    direct_transfers += 1
                else:
                    new_weights[g_id] = (w_1, s_id_1)
                    proposal_transfers += 1
            elif zero_shot_data.ndim == 1:
                if zero_shot_data[..., 2] > one_shot_data[..., 2]:
                    new_weights[g_id] = (w_0, s_id_0)
                    direct_transfers += 1
                else:
                    new_weights[g_id] = (w_1, s_id_1)
                    proposal_transfers += 1

        self.direct_transfers[self.t] = direct_transfers
        self.proposal_transfers[self.t] = proposal_transfers

        # new weights are about to be assigned, so
        # I don't think it's worth reseting the weights to the zero_state
        return new_weights
