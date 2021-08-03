from ray.rllib.utils.annotations import override
from ray.tune import Trainable


class ResetConfigOverride:
    @override(Trainable)
    def reset_config(self, new_config):

        def update(env):
            _ = env.reset(level_string=new_config['env_config']['level_string'])

        # do something with new_config
        self.workers.foreach_env(update)
        return True  # <- signals successful reset


if __name__ == "__main__":
    import ray
    import os
    import sys

    from utils.register import Registrar
    from utils.loader import load_from_yaml
    from gym_factory import GridGameFactory
    from network_factory import NetworkFactory
    from generators.AIIDE_generator import EvolutionaryGenerator
    from solvers.SingleAgentSolver import SingleAgentSolver

    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.utils import add_mixins

    os.chdir('..')
    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1)

    args_file = os.path.join('args.yaml')
    args = load_from_yaml(args_file)

    registry = Registrar(file_args=args)


    # level_string = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''
    # generator = EvolutionaryGenerator(level_string, file_args=registry.get_generator_config)
    #
    # # print(registry)
    gf = GridGameFactory(env_name=registry.env_name, env_wrappers=[])
    nf = NetworkFactory(registry.network_name, registry.get_nn_build_info)
    # # print(registry.trainer_config['model'])
    sas = SingleAgentSolver.remote(trainer_constructor=registry.trainer_constr,
                                   trainer_config=registry.trainer_config,
                                   registered_gym_name=registry.env_name,
                                   network_factory=nf,
                                   gym_factory=gf)

    new_config = registry.get_trainer_config
    new_config['env_config']['level_string'] = '''wwwwwwwwwwwww\nw....+e.....w\nw...........w\nw..A........w\nw...........w\nw...........w\nw.....w.....w\nw.g.........w\nwwwwwwwwwwwww\n'''

    r1 = sas.update_lvl_in_trainer.remote(new_config)

    new_new_config = registry.get_trainer_config
    new_new_config['env_config']['level_string'] = '''wwwwwwwwwwwww\nwwwwwwwwwwwww\nwwwwwwwwwwwww\nw..A........w\nwwwwwwwwwwwww\nwwwwwwwwwwwww\nwwwwwwwwwwwww\nwwwwwwwwwwwww\nwwwwwwwwwwwww\n'''
    r2 = sas.update_lvl_in_trainer.remote(new_new_config)

    print(ray.get([r1, r2]))

    ray.shutdown()