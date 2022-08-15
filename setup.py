import os
from setuptools import setup

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name="levelup",
    version="0.2.1",
    author="Sebastian Junges",
    author_email="sjunges@cs.ru.nl",
    maintainer="Sebastian Junges",
    maintainer_email="sjunges@cs.ru.nl",
    url="",
    description="Prototype implementation for hierarchical MDP model checking",
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=["levelup"],
    install_requires=[
        "stormpy>=1.7.0"
    ],
    python_requires='>=3',
)
