import os
import ray
import torch
from solvers.base import BaseSolver
from evaluators.rollout import rollout
from griddly.util.rllib.environment.core import RLlibEnv
from utils.place_rllib_logger_util import custom_log_creator


@ray.remote
class SingleAgentSolver(BaseSolver):
    def __init__(self, trainer_constructor, trainer_config, registered_gym_name, network_factory, gym_factory, weights={},
                 log_id=0):
        BaseSolver.__init__(self)
        # todo We might want to name these agents to access via keys
        #  rather than a convention of [network].

        # todo switch to having factories in here?
        # self.agent = solver[0]
        self.key = 0
        self.exp = log_id.split('_')[0]
        self.log_id = log_id
        self.trainer_config = trainer_config
        self.registered_gym_name = registered_gym_name
        self.network_factory = network_factory
        self.gym_factory = gym_factory
        self.trainer = trainer_constructor(config=trainer_config, env=registered_gym_name,
                                           logger_creator=custom_log_creator(os.path.join('..', 'enigma_logs', self.exp),
                                                                             f'POET_{log_id}.')
                                           )
        self.agent = network_factory.make()(weights)
        self.env = gym_factory.make()(trainer_config['env_config'])
        if bool(weights):
            self.set_weights(weights)

    @ray.method(num_returns=1)
    def evaluate(self, env_config, solver_id, generator_id) -> dict:
        """Run one rollout of the given actor(s) in the given env

        :param env_generator_fn: fn to generate RLlibEnv environment to run the simulation
        :param env_config: dict of generator details
        :param solver_id: id of solver being evaluated
        :param generator_id: id of generator the solver is being evaluated in
        :return: result information e.g. final score, win_status, etc.
        """

        _ = self.env.reset(level_string=env_config['level_string'])
        info, states, actions, rewards, win, logps, entropies = rollout(self.agent, self.env, 'cpu')
        return_kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps,
                         'entropy': entropies}

        return {self.key: {"info": info, "score": sum(rewards), "win": win == 'Win', 'kwargs': return_kwargs},
                'solver_id': solver_id, 'generator_id': generator_id}

    @ray.method(num_returns=1)
    def optimize(self, trainer_config, level_string_monad, **kwargs):
        """Run one step of optimization!! Update local agent

        :param trainer_config: config dict for e.g. PPO.
        :param level_string_monad:  callback to allow for dynamically created strings
        :return: dict of {optimized weights, result_dict}
        """

        trainer_config['env_config']['level_string'], _ = level_string_monad()
        is_updated = self.update_lvl_in_trainer(trainer_config)
        result = self.trainer.train()
        self.trainer.log_result(result)
        self._update_local_agent(self.get_weights())

        del result['config'] # Remove large and unserializable config

        return {self.key: {'weights': self.agent.state_dict(),
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

    def update_lvl_in_trainer(self, config_with_new_level):
        self.trainer_config = config_with_new_level
        return self.trainer.reset_config(config_with_new_level)

    def _update_local_agent(self, weights):
        self.agent.load_state_dict(weights)

    def get_key(self):
        return self.key

    def get_agent(self):
        return self.agent

    def value_function(self, level_string):
        state = self.env.reset(level_string=level_string)
        logits, h_state = self.agent.forward({'obs': torch.FloatTensor([state])}, [0], 1)
        return self.agent.value_function().item()

    def get_weights(self) -> dict:
        weights = self.trainer.get_weights()
        tensor_weights = {k: torch.FloatTensor(v) for k, v in weights['default_policy'].items()}
        return tensor_weights

    def set_weights(self, new_weights: dict):
        self.trainer.set_weights(weights={'default_policy': new_weights})
        self._update_local_agent(new_weights)

    def release(self):
        self.env.game.release()
        ray.actor.exit_actor()

    @ray.method(num_returns=1)
    def device(self):
        return next(self.trainer.get_policy().model.parameters()).device

    @ray.method(num_returns=1)
    def get_picklable_state(self):
        return {
            'trainer_config': self.trainer_config,
            'registered_gym_name': self.registered_gym_name,
            'network_factory': self.network_factory,
            'gym_factory': self.gym_factory,
            'weights': self.get_weights(),
            'log_id': self.log_id
        }
