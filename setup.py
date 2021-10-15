from setuptools import setup
from setuptools import find_packages

setup(name='gcc',
      description='Efficient Graph Convolutional Representation Learning for Graph Clustering.',
      install_requires=[
            "numpy~=1.19.5",
            "tensorflow~=2.4.1",
            "scipy~=1.5.3",
            "pandas~=0.25.1",
            "scikit-learn~=0.22",
            "setuptools~=41.4.0"
      ],
      package_data={'gcc': ['README.md']},
      packages=find_packages())
