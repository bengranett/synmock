#!/usr/bin/env python

from setuptools import setup

setup(name='synmock',
      version='0.0.3',
      description='LognormCat',
      author='Ben Granett',
      author_email='granett@gmail.com',
      packages=['synmock', 'synmock'],
      entry_points = {
        'console_scripts': ['synmock=synmock.main:main'],
    }
 )