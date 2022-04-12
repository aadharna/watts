import ray
import torch
import numpy as np
from collections import namedtuple

Rollout_results = namedtuple('Rollout_results',
                             ['info', 'states', 'values', 'actions', 'rewards', 'win', 'logps', 'entropies', 'dones', 'net_info'])


def rollout(actor, env) -> Rollout_results:
    """Run a rollout on a given actor and environment.

    :param actor: RLlib Policy Class: You can build this using the NetworkFactory
    :param env: An RLlibEnv (OpenAI Env that interfaces with RLlib) to evaluate the solver in
    :return: evaluation result state
    """
    state = env.reset()
    done = False

    rewards = []
    values = []
    states = []
    actions = []
    logps = []
    entropies = []
    dones = []
    network_infos = []
    win = False

    while not done:
        action, unbatched_state, network_info = actor.compute_single_action(state, full_fetch=True)
        logp = network_info.get('action_logp', 0)
        value = network_info.get('vf_preds', 0)
        try:
            # need to add a batch dim the distribution inputs.
            entropy = actor.dist_class(np.expand_dims(network_info['action_dist_inputs'], 0), actor.model).entropy().mean().item()
        except KeyError as e:
            entropy = 0
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        values.append(value)
        rewards.append(reward)
        logps.append(logp)
        entropies.append(entropy)
        dones.append(not done)
        network_infos.append(network_info)
        state = next_state

    if "PlayerResults" in info:
        win = info['PlayerResults']['1']

    results = Rollout_results(info=info,
                              states=states,
                              values=values,
                              actions=actions,
                              rewards=rewards,
                              win=win,
                              logps=logps,
                              entropies=entropies,
                              dones=dones,
                              net_info=network_infos)
    return results


@ray.remote
class RemoteRolloutActor:
    def __init__(self, network_factory, env_factory, env_config):
        self.nn_make_fn = network_factory.make()
        self.env_make_fn = env_factory.make()
        self.env_config = env_config
        self.env = self.env_make_fn(env_config)
        self.agent = self.nn_make_fn({})

    def run_rollout(self, nn_weights, env_config):
        self.agent.model.load_state_dict(nn_weights)
        if 'level_string' in env_config:
            _ = self.env.reset(level_string=env_config['level_string'])
        elif 'level_id' in env_config and 'level_string' not in env_config:
            _ = self.env.reset(level_id=env_config['level_id'])
        result = rollout(self.agent, self.env)
        return result


if __name__ == '__main__':
    import os
    import ray
    from ray.util import ActorPool
    from watts.utils.loader import load_from_yaml
    from watts.utils.register import Registrar
    from watts.utils.gym_wrappers import add_wrappers
    from watts.gym_factory import WalkerFactory, GridGameFactory
    from watts.network_factory import NetworkFactory
    from watts.generators.AIIDE_generator import EvolutionaryGenerator
    from watts.generators.WalkerConfigGenerator import WalkerConfigGenerator
    from watts.utils.box2d.biped_walker_custom import DEFAULT_ENV
    while 'poet_distributed.py' not in os.listdir('.'):
        os.chdir('..')
    print(os.listdir('.'))

    ray.init(local_mode=True)

    args = load_from_yaml(fpath=os.path.join('sample_args', 'args.yaml'))
    args.exp_name = 'remotetest'
    args.opt_algo = "PPO"

    registry = Registrar(file_args=args)
    # game_schema = GameSchema(registry.gdy_file) # Used for GraphValidator
    wrappers = add_wrappers(args.wrappers)
    gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)
    # gym_factory = WalkerFactory(registry.env_name, env_wrappers=wrappers)
    network_factory = NetworkFactory(registry.network_name, registry.get_nn_build_info,
                                     registry.policy_class)
    generator = EvolutionaryGenerator(args.initial_level_string,
                                      file_args=registry.get_generator_config)
    # walker_generator = WalkerConfigGenerator(parent_env_config=DEFAULT_ENV)

    env = gym_factory.make()(registry.get_config_to_build_rllib_env)
    nn = network_factory.make()({})
    remoteRolloutActors = [RemoteRolloutActor.remote(network_factory=network_factory,
                                                     env_factory=gym_factory,
                                                     env_config=registry.get_config_to_build_rllib_env) for _ in range(2)]
    pool = ActorPool(actors=remoteRolloutActors)
    g2 = generator.mutate()
    env_config = registry.get_config_to_build_rllib_env
    env_config['level_string'], _ = g2.generate_fn_wrapper()()
    results = list(pool.map(lambda a, v: a.run_rollout.remote(**v), [{'nn_weights':nn.model.state_dict(),
                                                                      'env_config': env_config},
                                                                    {'nn_weights': nn.model.state_dict(),
                                                                     'env_config': env_config}]))

    print(results[0].rewards)

    ray.shutdown()
