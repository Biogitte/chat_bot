#!/usr/bin/python3
from setuptools import find_packages, setup

setup(
    name='src',
    package_dir={'src': 'src'},
    packages=find_packages(),
    version='0.1.0',
    description='First Aid Chatbot',
    author='Birgitte Nilsson',
    license='MIT')

find_packages()
