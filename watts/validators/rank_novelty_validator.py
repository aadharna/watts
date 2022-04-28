import numpy as np
from typing import Tuple, Dict, List

import ray
from ray.util import ActorPool
from ray.rllib.agents.es.utils import compute_centered_ranks

from watts.evaluators.rollout import RemoteRolloutActor
from watts.generators.base import BaseGenerator
from watts.solvers.base import BaseSolver
from watts.validators.level_validator import LevelValidator


class RankNoveltyValidator(LevelValidator):
    """This is the method by which Wang et al determine the novelty of an environment
    https://arxiv.org/abs/2003.085363 see page 4.
    For their code, see: https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/es.py#L586
    This has been adapted to watts from the uber-ai code so it will not look 1-1, but the behaviours are replicated.

    # TODO
    Method to fix the "we can't import torch too many times" issue that this causes:
    Create an actor pool in here that can just live and run the evaluations.
    Basically, this method should get a few remote processes of its own where it can run evaluates
    This is for the archived solutions that don't have their own processes anymore which we can send work to
    """
    def __init__(self, density_threshold, env_config, historical_archive,
                 agent_factory, env_factory,
                 k=5, low_cutoff: float = -np.inf, high_cutoff: float = np.inf):
        """

        @param density_threshold: Scaler for determining if the new generator is novel based on its average distance to its nearest neighbors
        @param env_config: config to load level information into
        @param historical_archive: the archive that contains all agent-generator/env pairs that are no longer active
        @param agent_factory: factory to create new rllib policy NNs
        @param env_factory: factory to create new learning environments
        @param k: how many neighbors should I calculate my distance to?
        @param low_cutoff: Low cutoff used in normalizing the ranking scheme
        @param high_cutoff: High cutoff used in normalizing the ranking scheme
        """
        self.density_threshold = density_threshold
        self.k = k
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff
        self.env_config = env_config
        # this is a pointer to a dict that contains serialized
        # versions of the PAIR objects
        self.historical_archive = historical_archive
        self.pata_ecs = {}

        self.rollout_actor_pointers = [RemoteRolloutActor.remote(network_factory=agent_factory,
                                                                 env_factory=env_factory,
                                                                 env_config=self.env_config) for _ in range(5)]
        self.actor_pool = ActorPool(actors=self.rollout_actor_pointers)

    def validate_level(self, generators: List[BaseGenerator], solvers: List[BaseSolver], **kwargs) -> Tuple[bool, Dict]:
        """

        @param generators: Generator class that we can extract a level string from
        @param solvers: Solver class that can play a game
        @param kwargs: future proofing
        @return: True/False, distance data
        """
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
        """

        @param generators: Generator class that we can extract a level string from
        @param solvers: Solver class that can play a game
        @param kwargs: future proofing
        @return:
        """
        def cap_score(score, lower, upper):
            if score < lower:
                score = lower
            elif score > upper:
                score = upper

            return score

        raw_scores = []
        pooled_work = []
        self.env_config['level_string'], _ = generator.generate_fn_wrapper()()
        for archived_pair_id, archived_pair in self.historical_archive.items():
            pooled_work.append({
                'nn_weights': archived_pair['solver']['weights'],
                'env_config': self.env_config
            })

        for i, source_optim in enumerate(solvers):
            pooled_work.append({
                'nn_weights': ray.get(source_optim.get_weights.remote()),
                'env_config': self.env_config
            })

        # this list cast returns actual answers and not refs that still need to be collected
        result_list = list(self.actor_pool.map(lambda a, v: a.run_rollout.remote(**v), pooled_work))

        for r in result_list:
            raw_scores.append(cap_score(sum(r.rewards), lower=self.low_cutoff, upper=self.high_cutoff))

        pata_ec = compute_centered_ranks(np.array(raw_scores))
        self.pata_ecs[generator.id] = pata_ec


def euclidean_distance(x, y):
    """euclidean distance calculation from:
    https://github.com/uber-research/poet/blob/8669a17e6958f80cd547b2de61c51d4518c833d9/poet_distributed/novelty.py#L30

    @param x: nd-vector scores on d levels
    @param y: nd-vector scores on d levels
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
