#! /usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
        name='pyihm',
        version='2.0.0',
        author='Francesco Bruno, Letizia Fiorucci',
        author_email='bruno@cerm.unifi.it',
        description='Indirect Hard Modelling, in Python',
        url='https://github.com/MetallerTM/pyihm',
        long_description=long_description,
        long_description_content_type = 'text/markdown',
        classifiers = [
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            'License :: OSI Approved :: BSD License'
            ],
        license='LICENSE.txt',
        install_requires = ['numpy', 'scipy', 'lmfit', 'seaborn', 'nmrglue', 'matplotlib>=3.7', 'csaps', 'klassez>=0.4a.6'],
        packages=['pyihm'],
        include_package_data = True,
        python_requires = '>=3.10',
        )
