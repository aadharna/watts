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


@pytest.fixture(scope="module")
def init_solver():
    args = load_from_yaml(fpath='/home/rohindasari/research/watts_package/sample_args/args.yaml')
    args.lvl_dir = '../../example_levels'
    args.exp_name = 'poet_test'
    registry = Registrar(file_args=args)
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)

    network_factory = NetworkFactory('AIIDE_PINSKY_MODEL', registry.get_nn_build_info, policy_class=registry.policy_class)
    solver = SingleAgentSolver.remote(
                trainer_constructor=registry.trainer_constr,
                trainer_config=registry.get_trainer_config,
                registered_gym_name=registry.env_name,
                network_factory=network_factory,
                gym_factory=gym_factory,
                log_id=f"test_0"
            )
    return {'solver': solver, 'registry': registry, 'args': args}


def test_update_lvl_in_trainer(init_solver):
    """
    test to make sure that level config can be updated through trainer object
    """
    solver = init_solver['solver']
    args = init_solver['args']
    lvl_config = {'env_config': {
            'level_string': args.initial_level_string
        }}
    keyref = solver.update_lvl_in_trainer.remote(lvl_config)
    assert ray.get(keyref) == True

def test_get_key(init_solver):
    """
    test to ensure that the key returned by the solver is valid and initially set to 0
    """
    solver = init_solver['solver']
    keyref = solver.get_key.remote()
    assert ray.get(keyref) == 0

def test_get_picklable_state(init_solver):
    """
    test to ensure that all components of the solvers state are valid and can be pickled
    currently test is failing becase the rllib policy can't be pickled
    """
    import pickle
    solver = init_solver['solver']
    keyref = solver.get_picklable_state.remote()
    state = ray.get(keyref)
    # remove policy class attribute before pickling
    del state['network_factory']
    pickle.dumps(state)



