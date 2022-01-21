import numpy as np
from typing import Tuple, Dict, List

import ray
from ray.rllib.agents.es.utils import compute_centered_ranks

from .level_validator import LevelValidator
from ..generators.base import BaseGenerator
from ..solvers.base import BaseSolver


class RankNoveltyValidator(LevelValidator):
    """This is the method by which Wang et al determine the novelty of an environment
    https://arxiv.org/abs/2003.085363 see page 4.
    For their code, see: https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/es.py#L586
    This has been adapted to watts from the uber-ai code so it will not look 1-1, but the behaviours are replicated.


    TODO: Note, I think that this particular class wants access to the PAIR objects. Therefore it should be something else than
    a Validator even though this should also fit into that paradigm.
    We will call this class at the start of the POETStrategy::_get_child_list fn to update the agent rankings.
    """
    def __init__(self, density_threshold, env_config, historical_archive, k=5, low_cutoff: float = -np.inf, high_cutoff: float = np.inf):
        self.density_threshold = density_threshold
        self.k = k
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.env_config = env_config
        # this is a pointer to a dict that contains serialized
        # versions of the PAIR objects
        self.historical_archive = historical_archive
        self.pata_ecs = {}

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        return False, {'top_k_mean': 0}
        # distances = []
        # generator_pata_ec = self.calculate_pata_ec(generators[0], solvers)
        # for point in self.historical_archive.values():
        #     distances.append(euclidean_distance(self.pata_ecs[point['pair_id']], generator_pata_ec))
        #
        # for point in solvers:
        #     distances.append(euclidean_distance(point.pata_ec, generator_pata_ec))
        #
        # # Pick k nearest neighbors
        # distances = np.array(distances)
        # top_k_indicies = (distances).argsort()[:self.k]
        # top_k = distances[top_k_indicies]
        # # calculate the average distance to the kNNs.
        # top_k_mean = top_k.mean()
        # return top_k_mean > self.density_threshold, {'top_k_mean': top_k_mean}

    def calculate_pata_ec(self, generator: BaseGenerator, solvers: List[BaseSolver], **kwargs):
        """Should this function be somewhere else and then it is called on a PAIR object where we pass in
        all the weights we want to try? """
        def cap_score(score, lower, upper):
            if score < lower:
                score = lower
            elif score > upper:
                score = upper

            return score

        raw_scores = []
        refs = {}
        self.env_config['level_string'] = generator.generate_fn_wrapper()()
        for i, source_optim in enumerate(solvers):
            result_ref = source_optim.evaluate.remote(self.env_config, solver_id=0, generator_id=0)
            key_ref = source_optim.get_key.remote()
            refs[i] = {'res': result_ref, 'key_ref': key_ref}

        results = [v['res'] for k, v in refs.items()]
        keys = [v['key_ref'] for k, v in refs.items()]
        results = ray.get(results)
        keys = ray.get(keys)

        for result, key in zip(results, keys):
            raw_scores.append(cap_score(result[key]['score'], lower=self.low_cutoff, upper=self.high_cutoff))

        pata_ec = compute_centered_ranks(np.array(raw_scores))
        self.pata_ecs[generator.id] = pata_ec
        return pata_ec


def euclidean_distance(x, y):
    """euclidean distance calculation from:
    https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/novelty.py#L30

    :param x:
    :param y:
    :return:
    """
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)
