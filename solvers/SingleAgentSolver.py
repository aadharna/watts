import ray
from solvers.base import BaseSolver
from evaluators.rollout import rollout
from griddly.util.rllib.environment.core import RLlibEnv


class SingleAgentSolver(BaseSolver):
    def __init__(self, solver):
        BaseSolver.__init__(self)
        # todo We might want to name these agents to access via keys
        #  rather than a convention of [network].

        # todo switch to having factories in here?
        self.agent = solver[0]
        self.key = 0

    def evaluate(self, env: RLlibEnv) -> dict:
        """Run one rollout of the given actor(s) in the given env

        :param env: RLlibEnv environment to run the simulation
        :return: result information e.g. final score, win_status, etc.
        """
        info, states, actions, rewards, win, logps, entropies = rollout(self.agent, env)
        kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps, 'entropy': entropies}

        return {self.key: {"info": info, "score": sum(rewards), "win": win == 'Win', 'kwargs': kwargs}}

    def optimize(self, trainer_constructor, trainer_config, registered_gym_name, level_string_monad,
                 **kwargs):
        """Run one step of optimization!!

        :param trainer_constructor: constructor for algo to optimize wtih e.g. ppo.PPOTrainer for rllib to run optimization.
        :param trainer_config: config dict for e.g. PPO.
        :param registered_gym_name: name of env registered with ray via `env_register`
        :param level_string_monad:  callback to allow for dynamically created strings
        :param network_weights: torch state_dict
        :return: dict of {optimized weights, result_dict}
        """

        # todo same as rollout.py
        # todo will probably have to change this to first instantiate a generator model
        # and then query it for the levels.
        #  That will allow something like PAIRED to function?
        trainer_config['env_config']['level_string'], _ = level_string_monad()
        trainer = trainer_constructor(config=trainer_config, env=registered_gym_name)
        trainer.get_policy().model.load_state_dict(self.agent.state_dict())
        result = trainer.train()

        return {self.key: {'weights': trainer.get_policy().model.state_dict(),
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

    def get_weights(self) -> dict:
        return {self.key: self.agent.state_dict()}

    def set_weights(self, new_weights: list):
        self.agent.load_state_dict(new_weights[self.key])
