import numpy as np
from typing import Tuple, Dict, List

import ray
from ray.rllib.agents.es.utils import compute_centered_ranks

from watts.evaluators.rollout import remote_rollout
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


    Should this wrap a RankStrategy?? I think so. And then we can do the pata_ec calculations
    in the rank strategy module and do the knn portion here in the validator.
    I think this makes sense.
    """
    def __init__(self, density_threshold, env_config, historical_archive,
                 agent_make_fn, env_make_fn,
                 k=5, low_cutoff: float = -np.inf, high_cutoff: float = np.inf):
        self.density_threshold = density_threshold
        self.k = k
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.env_config = env_config
        # this is a pointer to a dict that contains serialized
        # versions of the PAIR objects
        self.historical_archive = historical_archive
        self.pata_ecs = {}
        self.agent_make_fn = agent_make_fn
        self.env_make_fn = env_make_fn

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        proposed_generator = kwargs.get('proposed_generator', None)
        if proposed_generator is None:
            raise ValueError('The RankNoveltyValidator needs to know what the proposed '
                             'new generator is as well as receive all of the active_population info.')

        self.calculate_pata_ec(proposed_generator, solvers=solvers)

        distances = []
        for p_id, pata_ec in self.pata_ecs.items():
            if not p_id == proposed_generator.id:
                distances.append(euclidean_distance(pata_ec, self.pata_ecs[proposed_generator.id]))

        distances = np.array(distances)
        top_k_indicies = (distances).argsort()[:self.k]
        top_k = distances[top_k_indicies]
        # calculate the average distance to the kNNs.
        top_k_mean = top_k.mean()
        return top_k_mean > self.density_threshold, {'top_k_mean': top_k_mean}

    def calculate_pata_ec(self, generator: BaseGenerator, solvers: List[BaseSolver], **kwargs):
        def cap_score(score, lower, upper):
            if score < lower:
                score = lower
            elif score > upper:
                score = upper

            return score

        refs = []
        raw_scores = []
        self.env_config['level_string'], _ = generator.generate_fn_wrapper()()
        for archived_pair_id, archived_pair in self.historical_archive.items():
            ref = remote_rollout.remote(self.agent_make_fn, self.env_make_fn,
                                        archived_pair['solver']['weights'],
                                        self.env_config)
            refs.append(ref)

        for i, source_optim in enumerate(solvers):
            source_weights = ray.get(source_optim.get_weights.remote())
            ref = remote_rollout.remote(self.agent_make_fn, self.env_make_fn,
                                        source_weights,
                                        self.env_config)
            refs.append(ref)

        result_list = ray.get(refs)

        for r in result_list:
            raw_scores.append(cap_score(sum(r.rewards), lower=self.low_cutoff, upper=self.high_cutoff))

        pata_ec = compute_centered_ranks(np.array(raw_scores))
        self.pata_ecs[generator.id] = pata_ec


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
