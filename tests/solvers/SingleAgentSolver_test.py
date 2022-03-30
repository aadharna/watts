import pytest
import ray
import sys
import os
import shutil

from watts.utils.loader import load_from_yaml
from watts.utils.register import Registrar
from watts.utils.gym_wrappers import add_wrappers
from watts.gym_factory import GridGameFactory
from watts.network_factory import NetworkFactory
from watts.solvers.SingleAgentSolver import SingleAgentSolver




@pytest.fixture(scope="session", autouse=True)
def init_ray_before_all_tests():

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)
    # taken from:
    # https://github.com/pynb-dag-runner/pynb-dag-runner/blob/development/workspace/pynb_dag_runner/conftest.py

    # - Init ray once before all tests.
    ray.init(
        num_cpus=1,
        # What's supposed to happen:
        # enable tracing and write traces to /tmp/spans/<pid>.txt in JSONL format
        # this argument throws the following error when initializing a remote object:
        # What's currently happening:
        # TypeError: got an unexpected keyword argument '_ray_trace_ctx'
        #_tracing_startup_hook="ray.util.tracing.setup_local_tmp_tracing:setup_tracing",
        )
    yield None
    ray.shutdown()


@pytest.fixture
def init_solver():
    args = load_from_yaml(fpath='/home/rohindasari/research/watts_package/sample_args/args.yaml')
    args.lvl_dir = '../../example_levels'
    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)

    network_factory = NetworkFactory('SimpleConvAgent', registry.get_nn_build_info)
    solver = SingleAgentSolver.remote(
                trainer_constructor=registry.trainer_constr,
                trainer_config=registry.get_trainer_config,
                registered_gym_name=registry.env_name,
                network_factory=network_factory,
                gym_factory=gym_factory,
                log_id=f"test_0"
            )
    return {'solver': solver, 'registry': registry}


def test_evaluate(init_solver):
    solver = init_solver['solver']
    keyref = solver.get_key.remote()
    print(ray.get(keyref))


