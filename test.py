from gym_factory import GridGameFactory
from utils.register import Registrar
from utils.gym_wrappers import add_wrappers
from utils.loader import load_from_yaml


args = load_from_yaml('forager_args.yaml')

registry = Registrar(file_args=args)
wrappers = add_wrappers(args.wrappers)
gym_factory = GridGameFactory(registry.env_name, env_wrappers=wrappers)


