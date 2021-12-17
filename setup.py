from distutils.core import setup

setup(
    name='watts',
    description='A framework for curriculum learning for reinforcement learning algorithms',
    author='Aaron Dharna <aadharna@gmail.com>, Charlie Summers <charlie@gomerits.com>, Rohin Dasari <rd2893@nyu.edu>',
    url='https://github.com/aadharna/watts',
    packages=['watts'],
    python_requires='>=3.7.10',
    install_requires=[
            'ray[default]',
            'ray[rllib]',
            'griddly',
            'networkx'
        ]
)

