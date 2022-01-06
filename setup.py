from distutils.core import setup

setup(
    name='watts',
    version='0.0.1',
    description='A framework for exploring open-endedness in reinforcement learning',
    author='Aaron Dharna <aadharna@gmail.com>, Charlie Summers <cgs2161@columbia.edu>, Rohin Dasari <rd2893@nyu.edu>',
    url='https://github.com/aadharna/watts',
    packages=['watts'],
    python_requires='>=3.7.10',
    install_requires=[
            'ray[all]==1.6.0',
            'griddly',
            'networkx',
            'pytest'
        ]
)

