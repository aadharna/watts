import os
import ray
from random import sample
from typing import Dict, List, Tuple

from generators.base import BaseGenerator
from gym_factory import GridGameFactory
from managers.base import Manager
from network_factory import NetworkFactory
from solvers.SingleAgentSolver import SingleAgentSolver
from utils.register import Registrar
from utils.loader import save_obj
from validators.level_validator import LevelValidator


class UniformLevelBuffer:
    def __init__(self, seed=None):
        self.buffer = set()
        if seed is not None:
            self.buffer.add(seed)

    def add(self, other):
        self.buffer.add(other)

    def remove(self, generator):
        if len(self.buffer) < 2:
            return
        self.buffer.remove(generator)

    def sample(self) -> BaseGenerator:
        assert len(self.buffer) > 0
        return sample(self.buffer, 1)[0]


class MCManager(Manager):
    def __init__(
            self,
            exp_name: str,
            reproduction_limit: int,
            mutation_timer: int,
            n_children: int,
            snapshot_timer: int,
            mutation_rate: float,
            agent: SingleAgentSolver,
            generator: BaseGenerator,
            validator: LevelValidator,
            gym_factory: GridGameFactory,
            network_factory: NetworkFactory,
            registrar: Registrar,
    ):
        """Minimal Criteria runner. See deepmind OEL paper page 40-41

        :param exp_name: exp_name from launch script
        :param gym_factory: factory to make new gym.Envs
        :param network_factory: factory to make new NNs
        :param registrar: class that dispenses necessary information e.g. num_poet_loops
        """
        super().__init__(exp_name, gym_factory, network_factory, registrar)
        self.agent = agent
        self.generator = generator
        self.validator = validator

        self.snapshot_timer = snapshot_timer
        self.mutation_timer = mutation_timer
        self.n_children = n_children

        self.stats = {}
        self.stats['lineage'] = []
        self.i = 1
        self.n_potentials = 1
        self.n_lvls = 1

        self.train_buffer = UniformLevelBuffer(generator)
        self.evaluate_buffer = UniformLevelBuffer()
        self.results = []

        self.reproduction_limit = reproduction_limit
        self.mutation_rate = mutation_rate

        self.cycle_out_counter = iter(range(999999999))
        self.reject_counter = iter(range(999999999))
        self.record = {}
        self.rejected = {}

    def new_generator(self):
        lineage = []
        for k in range(self.n_children):
            self.n_potentials += 1
            g = self.train_buffer.sample()
            child = g.mutate(mutation_rate=self.mutation_rate)
            self.evaluate_buffer.add(child)
            lineage.append((g.id, child.id))
        return lineage

    def test(self):
        """run the network on a set of test levels

        :return:
        """
        for c in range(5):
            reset_info = {'level_id': c}
            self.agent.test.remote(env_config=reset_info, test_lvl_id=c, step=self.i)
        return

    def evaluate(self) -> list:
        """run the network on a set of test levels

        :return:
        """
        r = []
        # for i in range(5):
        #     reset_info = {'level_id': i}
        #     ref = self.agent.evaluate.remote(env_config=reset_info, solver_id=0, generator_id=i)
        #     r.append(ref)
        # test_results = ray.get(r)
        # for tr in test_results:
        #     test_lvl_id = tr['generator_id']
        #     self.test_scores[test_lvl_id].append(tr[0])
        return r

    def optimize(self) -> list:
        g = self.train_buffer.sample()
        trainer_config = self.registrar.get_trainer_config
        level_string_monad = g.generate_fn_wrapper()
        ref = self.agent.optimize.remote(trainer_config=trainer_config,
                                         level_string_monad=level_string_monad)
        return [ref]

    def update_buffers(self):
        train_remove = []
        eval_remove = []
        for g in self.train_buffer.buffer:
            # if level is no longer valid:
            #    tag it, save it, remove it.
            is_valid, _ = self.validator.validate_level([g], [self.agent])
            # if level is no longer valid, remove it!
            if not is_valid:
                self.record[(next(self.cycle_out_counter), self.i)] = str(g)
                train_remove.append(g)

        for g in self.evaluate_buffer.buffer:
            # if generator is a valid generator to train on, add it to the training set
            is_valid, _ = self.validator.validate_level([g], [self.agent])
            if is_valid:
                self.train_buffer.add(g)
                self.n_lvls += 1
                eval_remove.append(g)
            # else, reject the level
            else:
                self.rejected[(next(self.reject_counter), self.i)] = str(g)
                eval_remove.append(g)

        self.agent.write.remote('n_removed_tasks', len(train_remove), self.i)
        # empty the sets at the end of the testing phase
        for g in eval_remove:
            self.evaluate_buffer.remove(g)
        for g in train_remove:
            self.train_buffer.remove(g)

        self.agent.write.remote('train_buffer_size', len(self.train_buffer.buffer), self.i)
        self.agent.write.remote('n_potentials', self.n_potentials, self.i)

    def run(self):
        """

        :return:
        """
        while self.n_potentials <= self.reproduction_limit:
            # print(f'loop {self.i}')
            print(f'levels created: {self.n_potentials} / {self.reproduction_limit}')
            i = self.i
            ref = self.optimize()
            self.results.append(ray.get(ref))

            if i % self.mutation_timer == 0:
                family_branch = self.new_generator()
                self.stats['lineage'].extend(family_branch)
                self.update_buffers()

            self.test()

            self.i += 1
            if i % self.snapshot_timer == 0:
                foo = {
                    'run_stats': self.stats,
                    'solver': ray.get(self.agent.get_picklable_state.remote()),
                    'trained_on': self.record
                }
                # manager.agent.release.remote()
                save_obj(foo,
                         os.path.join('../..', 'enigma_logs', self.exp_name),
                         f'MC_total_serialized_alg.{self.i}')
                pass

        # save run stats
        self.stats['n_rejected_lvls'] = next(self.reject_counter) - 1
        self.stats['n_accepted_lvls'] = self.n_lvls
        self.stats['n_potential_lvls'] = self.n_potentials

        self.stats['solver'] = ray.get(self.agent.get_picklable_state.remote())
        self.stats['rejected'] = self.rejected
        self.stats['trained_on'] = self.record
        return
