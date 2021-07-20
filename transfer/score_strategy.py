from itertools import product

from evaluators.remote_evaluate import async_evaluate_agent_on_level
from optimization.remote_optimize import async_optimize_solver_on_env


class ScoreStrategy:
    def score(self, solvers, generators, id_map) -> list:
        raise NotImplementedError


class ZeroShotCartesian(ScoreStrategy):
    def __init__(self, gym_factory, config):
        self.gf = gym_factory
        self.env_config = config

    def score(self, solvers, generators, id_map) -> list:
        """run the evaluate function on the cartesian product of solvers and generators

        :param solvers: list of Solver objects
        #TODO: make this a callable that will generate the string via e.g. NN-weights for a generator network
        :param generators: list of callable string levels
        :param id_map: list of tuples of (solver_id, generator_id) also created with product (id, id)
        :return: list of refs (and list of ids) for the evaluated solver-generator pairs
        """

        refs = []
        for i, (s, g) in enumerate(product(solvers, generators)):
            # todo: I don't like this. We need a better way of getting the solver/generator ids.
            # this is the same to do as below.
            solver_id, generator_id = id_map[i]
            ref = async_evaluate_agent_on_level.remote(gym_factory_monad=self.gf.make(),
                                                       rllib_env_config=self.env_config,
                                                       level_string_monad=g.generate_fn_wrapper(),
                                                       evaluate_monad=s.evaluate,
                                                       solver_id=solver_id,
                                                       generator_id=generator_id)
            refs.append(ref)
        return refs


# class OneStepCartesian(ScoreStrategy):
#     def __init__(self, gym_factory, config):
#         self.gf = gym_factory
#         self.env_config = config
#
#     def score(self, solvers, generators, id_map) -> list:
#         """run the evaluate function on the cartesian product of solvers and generators
#
#         :param solvers: list of Solver objects
#         #TODO: make this a callable that will generate the string via e.g. NN-weights for a generator network
#         :param generators: list of callable string levels
#         :param id_map: list of tuples of (solver_id, generator_id) also created with product (id, id)
#         :return: list of refs (and list of ids) for the evaluated solver-generator pairs
#         """
#
#         refs = []
#         for i, (s, g) in enumerate(product(solvers, generators)):
#             # todo: I don't like this. We need a better way of getting the solver/generator ids.
#             # this is the same to do as below.
#             solver_id, generator_id = id_map[i]
#             ref = async_optimize_solver_on_env.remote(
#                                                     # trainer_constructor,
#                                                     # trainer_config,
#                                                     # registered_gym_name,
#                                                     # level_string_monad,
#                                                     # optimize_monad,
#                                                     # **kwargs)
#
#             refs.append(ref)
#         return refs