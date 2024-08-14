from setuptools import setup, find_packages

setup(
    name='hrsam',
    version='0.1.0',
    packages=find_packages(exclude=['mmdetection', 'mmsegmentation', 'selective_scan']),
    install_requires=[],
)
