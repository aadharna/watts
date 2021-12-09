import pytest
import shutil

import ray

import sys
sys.path.append('../../')

from utils.loader import load_from_yaml
from utils.register import Registrar
from utils.gym_wrappers import add_wrappers
from gym_factory import GridGameFactory
from network_factory import NetworkFactory
from solvers.SingleAgentSolver import SingleAgentSolver
from solvers.MultiAgentSolver import MultiAgentSolver

@pytest.fixture(scope="session", autouse=True)
def init_ray_before_all_tests():
    # taken from:
    # https://github.com/pynb-dag-runner/pynb-dag-runner/blob/development/workspace/pynb_dag_runner/conftest.py
    # Clean up any spans from previous runs
    shutil.rmtree("/tmp/spans", ignore_errors=True)

    # - Init ray once before all tests.
    ray.init(
        num_cpus=1,
        # enable tracing and write traces to /tmp/spans/<pid>.txt in JSONL format
        # this argument throws the following error when initializing a remote object:
        # TypeError: got an unexpected keyword argument '_ray_trace_ctx'
        #_tracing_startup_hook="ray.util.tracing.setup_local_tmp_tracing:setup_tracing",
        )


@pytest.fixture
def init_components():
    args = load_from_yaml(fpath='../../forager_args.yaml')
    registry = Registrar(file_args=args, base_path='../../')
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)

    network_factory = NetworkFactory(registry.get_nn_build_info)
    return registry, network_factory, gym_factory


def test_thing(init_ray_before_all_tests, init_components):
    registry, network_factory, gym_factory = init_components
    solver = MultiAgentSolver.remote(
                trainer_constructor=registry.trainer_constr,
                trainer_config=registry.get_trainer_config,
                registered_gym_name=registry.env_name,
                network_factory=network_factory,
                gym_factory=gym_factory,
                log_id=f"test_0"
            )
    solver.get_agents.remote()







