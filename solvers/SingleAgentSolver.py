import ray
import torch
from solvers.base import BaseSolver
from evaluators.rollout import rollout
from griddly.util.rllib.environment.core import RLlibEnv


@ray.remote
class SingleAgentSolver(BaseSolver):
    def __init__(self, trainer_constructor, trainer_config, registered_gym_name, network_factory, gym_factory, weights={}):
        BaseSolver.__init__(self)
        # todo We might want to name these agents to access via keys
        #  rather than a convention of [network].

        # todo switch to having factories in here?
        # self.agent = solver[0]
        self.key = 0
        self.trainer = trainer_constructor(config=trainer_config, env=registered_gym_name)
        self.agent = network_factory.make()(weights)
        self.env = gym_factory.make()(trainer_config['env_config'])

    @ray.method(num_returns=1)
    def evaluate(self, env_generator_fn, env_config, solver_id, generator_id) -> dict:
        """Run one rollout of the given actor(s) in the given env

        :param env_generator_fn: fn to generate RLlibEnv environment to run the simulation
        :param env_config: dict of generator details
        :param solver_id: id of solver being evaluated
        :param generator_id: id of generator the solver is being evaluated in
        :return: result information e.g. final score, win_status, etc.
        """

        _ = self.env.reset(level_string=env_config['level_string'])
        agent_weights = self.get_weights()
        self._update_local_agent(agent_weights)
        info, states, actions, rewards, win, logps, entropies = rollout(self.agent, self.env, 'cpu')
        return_kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps,
                         'entropy': entropies}

        return {self.key: {"info": info, "score": sum(rewards), "win": win == 'Win', 'kwargs': return_kwargs},
                'solver_id': solver_id, 'generator_id': generator_id}

    @ray.method(num_returns=1)
    def optimize(self, trainer_config, level_string_monad, **kwargs):
        """Run one step of optimization!!

        :param trainer_config: config dict for e.g. PPO.
        :param level_string_monad:  callback to allow for dynamically created strings
        :return: dict of {optimized weights, result_dict}
        """

        trainer_config['env_config']['level_string'], _ = level_string_monad()
        self.trainer.reset_config(trainer_config)
        result = self.trainer.train()
        self._update_local_agent(self.get_weights())

        return {self.key: {'weights': self.agent.state_dict(),
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

    def _update_local_agent(self, weights):
        self.agent.load_state_dict(weights)

    def get_weights(self) -> dict:
        weights = self.trainer.get_weights()
        tensor_weights = {k: torch.FloatTensor(v) for k, v in weights['default_policy'].items()}
        return tensor_weights

    def set_weights(self, new_weights: dict):
        self.trainer.set_weights(weights={'default_policy': new_weights})
        # self.agent.load_state_dict(new_weights[self.key])

    def release(self):
        ray.actor.exit_actor()


if __name__ == "__main__":
    import ray
    import os
    import sys

    from utils.register import Registrar
    from utils.loader import load_from_yaml
    from gym_factory import GridGameFactory
    from network_factory import NetworkFactory
    from generators.AIIDE_generator import EvolutionaryGenerator


    os.chdir('..')
    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1)

    args_file = os.path.join('args.yaml')
    args = load_from_yaml(args_file)

    registry = Registrar(file_args=args)

    level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    generator = EvolutionaryGenerator(level_string, file_args=registry.get_generator_config)

    # print(registry)
    gf = GridGameFactory(env_name=registry.env_name, env_wrappers=[])
    nf = NetworkFactory(registry.network_name, registry.get_nn_build_info)
    # print(registry.trainer_config['model'])
    sas = SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                   trainer_config=registry.trainer_config,
                                   registered_gym_name=registry.env_name,
                                   network_factory=nf)

    eval_res = sas.evaluate.remote(gf.make(), registry.trainer_config['env_config'])
    opt_res  = sas.optimize.remote(registry.trainer_config, generator.generate_fn_wrapper(), pair_id=4)

    eval_res = ray.get(eval_res)
    opt_res = ray.get(opt_res)
    w = sas.get_weights.remote()
    print(w[0]['default_policy'].keys())
    # m = sas.trainer.get_policy()
    # print(next(m.model.parameters()).device, "for solver class")

    # ray.shutdown()
