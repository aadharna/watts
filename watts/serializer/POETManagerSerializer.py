import copy
import pickle

import ray
from ray.rllib.utils import add_mixins

from watts.managers import POETManager
from watts.utils.trainer_reset import ResetConfigOverride
from watts.utils.register import get_default_trainer_config_constructor_and_policy_fn, Registrar
from watts.solvers.SingleAgentSolver import SingleAgentSolver


class POETManagerSerializer():
    """
    Serialize the POET experiment so that it can be saved and reloaded on demand.
    """
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
        delattr(manager.registrar, 'policy_class')
        delattr(manager.network_factory, 'policy_class')

        # Take out unserializable solver and replace it with the state necessary to recreate it
        for pair in manager.active_population:
            pair.solver_state = pair.get_picklable_state()['solver']
            delattr(pair, 'solver')

        attributes = self.__dict__
        attributes['manager'] = manager
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state

        # Recreate trainer_constr in the registrar
        _, trainer_constr, policy_class_fn = get_default_trainer_config_constructor_and_policy_fn(self.manager.registrar.file_args.opt_algo)
        if self.manager.registrar.file_args.custom_trainer_config_override:
            trainer_constr = add_mixins(trainer_constr, [ResetConfigOverride])
        self.manager.registrar.trainer_constr = trainer_constr
        self.manager.registrar.policy_class = policy_class_fn(self.manager.registrar.get_trainer_config)

        # Recreate the solvers
        for pair in self.manager.active_population:
            nf = pair.solver_state['network_factory']
            nf.policy_class = self.manager.registrar.policy_class
            pair.solver = SingleAgentSolver.remote(trainer_constructor=trainer_constr,
                                                   trainer_config=pair.solver_state['trainer_config'],
                                                   registered_gym_name=pair.solver_state['registered_gym_name'],
                                                   network_factory=nf,
                                                   gym_factory=pair.solver_state['gym_factory'],
                                                   weights=pair.solver_state['weights'],
                                                   log_id=f"{pair.solver_state['log_id']}")
            delattr(pair, 'solver_state')
