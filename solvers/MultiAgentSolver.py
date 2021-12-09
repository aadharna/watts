import os
import ray
import torch
from solvers.base import BaseSolver
from evaluators.rollout import rollout
from griddly.util.rllib.environment.core import RLlibEnv
from utils.place_rllib_logger_util import custom_log_creator


@ray.remote
class MultiAgentSolver(BaseSolver):
    def __init__(
            self,
            trainer_constructor,
            trainer_config,
            registered_gym_name,
            network_factory,
            gym_factory,
            weights={}, # this should be an array of weights to support weights for each agent
            log_id=0
            ):
        self.key = 0

        self.exp = log_id.split('_')[0]
        log_path = os.path.join('..', 'enigma_logs', self.exp)
        self.trainer = trainer_constructor(
                config=trainer_config,
                env=registered_gym_name,
                logger_creator=custom_log_creator(log_path, f'POET_{log_id}')
            )

        self.agents = self._init_agents(network_factory)
        self.env = gym_factory.make()(trainer_config['env_config'])

        # this logic exists in the single agent solver
        # but due to the current ambiguity of the weights variable in the multi-agent context
        # it is commented out
        # this can be re-evaluated once weights is better defined in the multi-agent context

        #if bool(weights):
        #    self.set_weights(weights)

    def _init_agents(self, network_factory):
        factories = network_factory.make()
        agents = {}
        for name, factory in factories.items():
            agent = factory() # define how to handle weights here
            agents[name] = agent
        return agents

    @ray.method(num_returns=1)
    def evaluate(self, env_config, solver_id, generator_id) -> dict:
        """
        """

        self.env.reset(level_string=env_config['level_string'])
        # modify rollout method to support multiple agents
        #info, states, actions, rewards, win, logps, entropies = rollout(self.agent, self.env, 'cpu')
        #return_kwargs = {'states': states, 'actions': actions, 'rewards': rewards, 'logprobs': logps,
        #                 'entropy': entropies}

        #return {self.key: {"info": info, "score": sum(rewards), "win": win == 'Win', 'kwargs': return_kwargs},
        #        'solver_id': solver_id, 'generator_id': generator_id}

    ray.method(num_returns=1)
    def optimize(self, trainer_config, level_string_monad, **kwargs):
        trainer_config['env_config']['level_string'] = level_string_monad()
        is_updated = self.update_lvl_in_trainet(trainer_config) # need to implement
        result = self.trainer.train() # update trainer for each model
        self.trainer.log_result(result)
        self._update_local_agent(self.get_weights())
        del result['config'] # Remove large and unserializable config
        return {self.key: {'weights': self.agent.state_dict(), # -> update to handle multiple agents in self.agents dict
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

        def update_lvl_in_trainer(self):
            self.trainer_config = config_with_new_level
            return self.trainer.reset_config(config_with_new_level)

    def _get_agent_states(self):
        states = {}
        for name, agent in self.agents:
            states[name] = agent.state_dict()
        return states

    def get_agents(self):
        #return self.agents
        #return self.agents.keys()
        return



