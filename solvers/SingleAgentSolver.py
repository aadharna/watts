import ray
import torch
from solvers.base import BaseSolver
from evaluators.rollout import rollout
from griddly.util.rllib.environment.core import RLlibEnv


# @ray.remote
class SingleAgentSolver(BaseSolver):
    def __init__(self, trainer_constructor, trainer_config, registered_gym_name, network_factory):
        BaseSolver.__init__(self)
        # todo We might want to name these agents to access via keys
        #  rather than a convention of [network].

        # todo switch to having factories in here?
        # self.agent = solver[0]
        self.key = 0
        self.trainer = trainer_constructor(config=trainer_config, env=registered_gym_name)
        self.agent = network_factory.make()({})

    def evaluate(self, env: RLlibEnv) -> dict:
        """Run one rollout of the given actor(s) in the given env

        :param env: RLlibEnv environment to run the simulation
        :return: result information e.g. final score, win_status, etc.
        """
        agent_weights = self.trainer.get_weights()
        self._update_local_agent(agent_weights)
        info, states, actions, rewards, win, logps, entropies = rollout(self.agent, env, 'cpu')
        kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps, 'entropy': entropies}

        return {self.key: {"info": info, "score": sum(rewards), "win": win == 'Win', 'kwargs': kwargs}}

    def optimize(self, trainer_config, level_string_monad, **kwargs):
        """Run one step of optimization!!

        :param trainer_config: config dict for e.g. PPO.
        :param level_string_monad:  callback to allow for dynamically created strings
        :return: dict of {optimized weights, result_dict}
        """

        trainer_config['env_config']['level_string'], _ = level_string_monad()
        self.trainer.reset_config(trainer_config)
        result = self.trainer.train()
        self._update_local_agent(self.trainer.get_weights())

        return {self.key: {'weights': self.agent.state_dict(),
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

    def _update_local_agent(self, weights):
        state_dict = {k: torch.FloatTensor(v) for k, v in weights['default_policy'].items()}
        self.agent.load_state_dict(state_dict)

    def get_weights(self) -> dict:
        return {self.key: self.trainer.get_weights()}

    def set_weights(self, new_weights: dict):
        self.trainer.set_weights(weights=new_weights)
        # self.agent.load_state_dict(new_weights[self.key])


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
    sas = SingleAgentSolver(trainer_constructor=registry.trainer_constr, trainer_config=registry.trainer_config,
                            registered_gym_name=registry.env_name, network_factory=nf)

    eval_res = sas.evaluate(gf.make()(registry.trainer_config['env_config']))
    opt_res  = sas.optimize(registry.trainer_config, generator.generate_fn_wrapper(), pair_id=4)
    m = sas.trainer.get_policy()
    print(next(m.model.parameters()).device, "for solver class")

    ray.shutdown()
