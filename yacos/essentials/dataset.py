"""
Copyright 2021 Anderson Faustino da Silva.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import urllib.request
import tarfile

from absl import logging as lg


class classproperty(property):
    """class property decorator."""

    def __get__(self, cls, owner):
        """Decorate."""
        return classmethod(self.fget).__get__(None, owner)()


class Dataset:
    """Static class to downloat datasets."""

    __version__ = '1.0.0'

    __url = 'www.csl.uem.br/repository/data'
    __benchmarks = ['AnghaBench',
                    'AnghaBench_WholeFiles',
                    'embench-iot',
                    'MiBench']
    __training_data = ['AnghaBench_llvm_instructions_15000B_10043S',
                       'AnghaBench_llvm_instructions_15000B_1290S',
                       'AnghaBench_llvm_instructions_1500B_1290S',
                       'AnghaBench_llvm_instructions_levels']

    @classproperty
    def benchmarks(cls):
        """Getter."""
        return cls.__benchmarks

    @classproperty
    def training_data(cls):
        """Getter."""
        return cls.__training_data

    @staticmethod
    def download_benchmark(benchmark):
        """Download benchmarks.

        Parameter
        ---------
        training_data : str
        """
        if benchmark not in Dataset.__benchmarks:
            lg.error('Benchmark {} does not exist.'.format(benchmark))
            sys.sys.exit(1)

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.exit(1)

        directory = os.path.join(top_dir, 'dataset/benchmarks')
        os.makedirs(directory, exist_ok=True)

        if not os.path.isdir(os.path.join(directory, benchmark)):
            archive_file = '{}.tar.xz'.format(benchmark)
            urllib.request.urlretrieve(Dataset.__url, archive_file)
            with tarfile.open(archive_file, 'r:xz') as f:
                f.extractall(directory)

    @staticmethod
    def download_training_data(training_data):
        """Download training data.

        Parameter
        ---------
        training_data : str
        """
        if training_data not in Dataset.__training_data:
            lg.error('Training data {} does not exist.'.format(training_data))
            sys.sys.exit(1)

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.exit(1)

        directory = os.path.join(top_dir, 'dataset/training_data')
        os.makedirs(directory, exist_ok=True)

        if not os.path.isdir(os.path.join(directory, training_data)):
            archive_file = '{}.tar.xz'.format(training_data)
            urllib.request.urlretrieve(Dataset.__url, archive_file)
            with tarfile.open(archive_file, 'r:xz') as f:
                f.extractall(directory)
