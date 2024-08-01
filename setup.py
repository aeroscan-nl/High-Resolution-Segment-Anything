from setuptools import setup, find_packages

setup(
    name='hrsam',
    version='0.1.0',
    packages=find_packages(include=['high-resolution-segment-anything', 'engine', 'models']),
    install_requires=[],
)
