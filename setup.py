# -*- coding: utf-8 -*-
"""Python setuptools configuration.

Author:
    Fangzhou Li - fzli@ucdavis.edu

"""

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='msap',
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',  # GitHub link.
    author='Fangzhou Li',
    author_email='fzli@ucdavis.edu',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
    ]
)
