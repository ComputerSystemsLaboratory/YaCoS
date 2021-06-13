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
    __benchmark = ['AnghaBench',
                   'embench-iot',
                   'MiBench']
    __training_data = ['llvm_instructions.15k_progs.10044_seqs',
                       'llvm_instructions.15k_1290_seqs',
                       'llvm_instructions.1500_progs.1290_seqs',
                       'llvm_instructions.levels']

    @classproperty
    def benchmark(cls):
        """Getter."""
        return cls.__benchmark

    @classproperty
    def training_data(cls):
        """Getter."""
        return cls.__training_data

    @staticmethod
    def download_benchmark(benchmark,
                           outdir):
        """Download benchmarks.

        Parameter
        ---------
        training_data : str

        outdir : str
            Output directory
        """
        if benchmark not in Dataset.__benchmark:
            lg.error('Benchmark {} does not exist.'.format(benchmark))
            sys.sys.exit(1)

        os.makedirs(outdir, exist_ok=True)

        if not os.path.isdir(os.path.join(outdir, benchmark)):
            archive_file = '{}.tar.xz'.format(benchmark)
            urllib.request.urlretrieve(Dataset.__url, archive_file)
            with tarfile.open(archive_file, 'r:xz') as f:
                f.extractall(outdir)

    @staticmethod
    def download_training_data(training_data,
                               outdir):
        """Download training data.

        Parameter
        ---------
        training_data : str

        outdir : str
            Output directory
        """
        if training_data not in Dataset.__training_data:
            lg.error('Training data {} does not exist.'.format(training_data))
            sys.sys.exit(1)

        os.makedirs(outdir, exist_ok=True)

        if not os.path.isdir(os.path.join(outdir, training_data)):
            archive_file = '{}.tar.xz'.format(training_data)
            urllib.request.urlretrieve(Dataset.__url, archive_file)
            with tarfile.open(archive_file, 'r:xz') as f:
                f.extractall(outdir)
