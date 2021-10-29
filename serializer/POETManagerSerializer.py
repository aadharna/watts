import copy
import pickle
import ray

from managers import POETManager
from ray.rllib.utils import add_mixins
from solvers.SingleAgentSolver import SingleAgentSolver
from utils.register import get_default_trainer_config_and_constructor, Registrar
from utils.trainer_reset import ResetConfigOverride


class POETManagerSerializer():
    def __init__(self, manager: POETManager):
        self.manager = manager

    def serialize(self):
        with open('snapshot.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize() -> POETManager:
        with open('snapshot.pkl', 'rb') as f:
            return pickle.load(f).manager

    def __getstate__(self):
        manager = copy.deepcopy(self.manager)

        # Take out registrar's unserializable trainer_constr - we have the information to recreate it from file_args
        delattr(manager.registrar, 'trainer_constr')

        # Take out unserializable solver and replace it with the state necessary to recreate it
        for pair in manager.active_population:
            pair.solver_state = ray.get(pair.solver.get_picklable_state.remote())
            delattr(pair, 'solver')

        attributes = self.__dict__
        attributes['manager'] = manager
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state

        # Recreate trainer_constr in the registrar
        _, trainer_constr = get_default_trainer_config_and_constructor(self.manager.registrar.file_args.opt_algo)
        if self.manager.registrar.file_args.custom_trainer_config_override:
            trainer_constr = add_mixins(trainer_constr, [ResetConfigOverride])
        self.manager.registrar.trainer_constr = trainer_constr

        # Recreate the solvers
        for pair in self.manager.active_population:
            pair.solver = SingleAgentSolver.remote(trainer_constructor=trainer_constr,
                                                   trainer_config=pair.solver_state['trainer_config'],
                                                   registered_gym_name=pair.solver_state['registered_gym_name'],
                                                   network_factory=pair.solver_state['network_factory'],
                                                   gym_factory=pair.solver_state['gym_factory'],
                                                   weights=pair.solver_state['weights'],
                                                   log_id=f"{pair.solver_state['log_id']}")
            delattr(pair, 'solver_state')
