# -*- coding: utf-8 -*-
"""Python setuptools configuration.

Author:
    Fangzhou Li - fzli@ucdavis.edu

"""

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='',
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',  # GitHub link.
    author='Fangzhou Li',
    author_email='fzli@ucdavis.edu',
    classifiers=[
        'Development Status :: 1 - Planning',
        # 'Environment ::',
        # 'Framework ::',
        # 'Intended Audience ::',
        # 'License ::',
        # 'Natural Language ::',
        # 'Operating System ::',
        # 'Programming Language ::',
        # 'Topic ::',
    ],
    keywords='',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
    ]
)
