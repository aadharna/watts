import os

import ray
import torch
from griddly.util.rllib.environment.core import RLlibEnv

from watts.solvers.base import BaseSolver
from watts.utils.place_rllib_logger_util import custom_log_creator
from watts.evaluators.rollout import rollout

from warnings import warn


@ray.remote
class SingleAgentSolver(BaseSolver):
    """This class is a ray.remote class. Therefore, when instantiated, it will be spun up on a different process.
    This class in general is responsible for defining how an agent is optimized and evaluated.
    """
    def __init__(self, trainer_constructor, trainer_config, registered_gym_name, network_factory, gym_factory, weights={},
                 log_id='foo_bar'):
        """This particular class is a generic wrapper into rllib's optimization suite.

        @param trainer_constructor: an rllib.trainer_constructor e.g. PPOTrainer
        @param trainer_config: The filled-in config dictionary necessary for the trainer_constructor above.
        @param registered_gym_name: the registered gym name of the learning env so that rllib can spin up new envs
        @param network_factory: The network factory so that we can create a new rllib pocliy NN
        @param gym_factory: The gym factor so that we can spin up a local env to evaluate in
        @param weights: Optional weights to set the neural to.
        @param log_id: Id for the logger to use. NEEDS a `_` in it.
        """
        BaseSolver.__init__(self)

        self.key = 0
        self.exp = log_id.split('_')[0]
        self.log_id = log_id
        self.trainer_config = trainer_config
        self.registered_gym_name = registered_gym_name
        self.network_factory = network_factory
        self.gym_factory = gym_factory
        self.trainer = trainer_constructor(config=trainer_config, env=registered_gym_name,
                                           logger_creator=custom_log_creator(os.path.join('.', 'watts_logs', self.exp),
                                                                             f'SAS_{log_id}.')
                                           )
        self.tensorboard_writer = self.trainer._result_logger._loggers[2]

        self.agent = network_factory.make()(weights)
        self.env = gym_factory.make()(trainer_config['env_config'])
        # record videos of the agents
        self.env.on_episode_start(worker_idx=self.log_id, env_idx=0)
        if bool(weights):
            self.set_weights(weights)

        self.warned = False

    @ray.method(num_returns=0)
    def test(self, env_config, test_lvl_id, step):
        """evaluate the NN on this passed in level and write the results to tensorboard

        @param env_config: config containing the level info
        @param test_lvl_id: test level_id for graph name
        @param step: x-axis value for tensorboard results graph
        @return:
        """
        result = self.evaluate(env_config, -1, -1)
        self.write(f'reward.test{test_lvl_id}', result[self.key]['score'], step)
        self.write(f'solved.test{test_lvl_id}', result[self.key]['win'], step)
        return

    @ray.method(num_returns=0)
    def write(self, name, value, step):
        """write this value to a tensorboard graph of this name with this x-value of step.

        @param name: which graph to write to
        @param value: the y-value of the datapoint
        @param step: the x-value of the datapoint
        @return:
        """
        self.tensorboard_writer.on_result({name: value, 'training_iteration': step})

    @ray.method(num_returns=1)
    def evaluate(self, env_config, solver_id, generator_id) -> dict:
        """Run one rollout of the given actor(s) in the given env

        NOTE: THIS WILL BE CHANGING VERY SOON so that instead of using a custom rollout function,
          this method will call the `evaluate` function built into rllib.

        @param env_config: dict of generator details
        @param solver_id: id of solver being evaluated
        @param generator_id: id of generator the solver is being evaluated in
        :return: result information e.g. final score, win_status, etc.
        """

        if not self.warned:
            warn('This method is deprecated.', DeprecationWarning, stacklevel=2)
            self.warned = True

        if 'level_string' in env_config:
            _ = self.env.reset(level_string=env_config['level_string'])
        elif 'level_id' in env_config and 'level_string' not in env_config:
            _ = self.env.reset(level_id=env_config['level_id'])
        else:
            raise ValueError('was not given a level to load into env')
        results = rollout(self.agent, self.env)
        return_kwargs = results._asdict()
        # foo = self.trainer.evaluate()

        return {
            self.key: {"info": results.info, "score": sum(results.rewards), "win": results.win == 'Win',
                       'kwargs': return_kwargs},
            'solver_id': solver_id,
            'generator_id': generator_id,
        }

    @ray.method(num_returns=1)
    def optimize(self, trainer_config, level_string_monad, **kwargs):
        """Run one step of optimization!! Update local agent

        @param trainer_config: config dict for e.g. PPO.
        @param level_string_monad:  callback to allow for dynamically created strings
        :return: dict of {optimized weights, result_dict}
        """

        trainer_config['env_config']['level_string'], _ = level_string_monad()
        is_updated = self.update_lvl_in_trainer(trainer_config)
        result = self.trainer.train()
        self.trainer.log_result(result)
        self._update_local_agent(self.get_weights())

        del result['config'] # Remove large and unserializable config

        return {self.key: {'weights': self.get_weights(),
                           "result_dict": result,
                           'pair_id': kwargs.get('pair_id', 0)
                           }
                }

    def update_lvl_in_trainer(self, config_with_new_level):
        """Update the level in the remote trainer object

        @param config_with_new_level: new config that contains the updated level
        @return: True if level is successfully laoded into the remote environemnt
        """
        self.trainer_config = config_with_new_level
        return self.trainer.reset_config(config_with_new_level)

    def _update_local_agent(self, weights):
        """Update the weights in the remote agent.

        @param weights: PyTorch Statedict of new weights to update the agent to.
        @return:
        """
        self.agent.model.load_state_dict(weights)

    def get_key(self):
        """
        get reference key to this agent's results. In this single-agent class, this will always be 0.
        @return:
        """
        return self.key

    def get_agent(self):
        """Get the agent class.

        todo I think this should probably be removed.
        @return:
        """
        return self.agent

    def value_function(self, level_string):
        """Get the agent's learned Value of a given level.

        @param level_string: level which is to query the agent's value function.
        @return: Scaler value.
        """
        state = self.env.reset(level_string=level_string)
        _, _, info = self.agent.compute_single_action(state)
        return info.get('vf_preds', 0)

    def get_weights(self) -> dict:
        """Get the weights from the remote trainer. Transform the weights into a stanrdard state-dict.

        @return: PyTorch loadable state-dict.
        """
        weights = self.trainer.get_weights()
        tensor_weights = {k: torch.FloatTensor(v) for k, v in weights['default_policy'].items()}
        return tensor_weights

    def set_weights(self, new_weights: dict):
        """Set the weights in the trainer class and local copy with these weights.

        @param new_weights: PyTorch state dict version of the weights.
        @return:
        """
        self.trainer.set_weights(weights={'default_policy': new_weights})
        self._update_local_agent(new_weights)

    def release(self):
        """Close the env and release any resources that this agent had claimed back to the OS.

        @return:
        """
        self.env.game.release()
        ray.actor.exit_actor()

    @ray.method(num_returns=1)
    def device(self):
        """Get what device this agent is being trained on by rllib.

        @return:
        """
        return next(self.trainer.get_policy().model.parameters()).device

    @ray.method(num_returns=1)
    def get_picklable_state(self):
        """Make the SAS class pickleable!

        @return: dict of SAS state. NOTE: the network_factory.policy_class still needs to be removed!
        """
        # del self.network_factory.policy_class # this is not serializable.
        return {
            'trainer_config': self.trainer_config,
            'registered_gym_name': self.registered_gym_name,
            'network_factory': self.network_factory,
            'gym_factory': self.gym_factory,
            'weights': self.get_weights(),
            'log_id': self.log_id
        }
