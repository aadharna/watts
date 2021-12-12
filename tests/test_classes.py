import gym
from griddly.util.rllib.environment.core import RLlibEnv
from watts.generators.base import BaseGenerator
from watts.solvers.base import BaseSolver
from watts.pair.agent_environment_pair import Pairing


class SimpleGymWrapper(gym.Wrapper, RLlibEnv):
    def __init__(self, env, env_config):
        gym.Wrapper.__init__(self, env=env)
        RLlibEnv.__init__(self, env_config=env_config)

        self.foo = 5


class MockGenerator(BaseGenerator):
    def __init__(self, expected_mutation_rate: float):
        super().__init__()
        self._expected_mutation_rate = expected_mutation_rate

    def mutate(self, mutation_rate: float):
        assert self._expected_mutation_rate == mutation_rate
        return self

    def update_from_lvl_string(self, level_string):
        raise NotImplementedError

    def generate_fn_wrapper(self):
        def _generate() -> str:
            raise NotImplementedError
        return _generate

    def __str__(self):
        raise NotImplementedError


class MockPair(Pairing):
    def __init__(self, solver, generator):
        super().__init__(solver, generator)

    def get_picklable_state(self):
        return {}


class MockSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    @property
    def evaluate(self, env_config, solver_id, generator_id):
        class Foo:
            def remote(self):
                return {0: {
                    'win': True,
                    'score': 0
                    }
                }
        return Foo()

    @property
    def get_key(self):
        class Foo:
            def remote(self):
                return 0
        return Foo()

    @property
    def set_weights(self, new_weights):
        class Foo:
            def remote(self):
                return
        return Foo()

    @property
    def get_weights(self) -> dict:
        class Foo:
            def remote(self):
                return {}
        return Foo()

    @property
    def release(self):
        class Foo:
            def remote(self):
                return
        return Foo()
