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
import csv
import glob
import tempfile
import subprocess
import numpy as np

from absl import logging as lg

from yacos.essentials import IO
from yacos.essentials import Engine
from yacos.essentials import Sequence
from yacos.info.ncc import llvm_ir_to_trainable


class Inst2Vec:
    """Inst2Vec Representation."""

    class classproperty(property):
        """classproperty decorator."""

        def __get__(self, cls, owner):
            """Decorate."""
            return classmethod(self.fget).__get__(None, owner)()

    __version__ = '1.0.0'

    __data_directory_ir = None
    __data_directory_seq = None

    __data_dir = 'yacos/inst2vec'
    __embeddings_file = 'inst2vec_augmented_embeddings.pickle'

    @staticmethod
    def prepare_benchmarks(benchmarks_filename,
                           benchmarks_directory):
        """We need to store all benchmarks in the same location.

        Parameters
        ----------
        benchmarks_filename : str

        benchmarks_directory: str
        """
        Inst2Vec.__data_directory_ir = tempfile.mkdtemp(suffix='_ir')

        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)

        for benchmark in benchmarks:
            index = benchmark.find('.')
            src = os.path.join(
                benchmarks_directory,
                benchmark[:index],
                benchmark[index+1:]
            )
            dst = os.path.join(Inst2Vec.__data_directory_ir, benchmark)

            cmdline = 'ln -s {} {}'.format(src, dst)
            try:
                subprocess.run(cmdline,
                               shell=True,
                               check=True,
                               capture_output=False)
            except subprocess.CalledProcessError:
                lg.error('Prepare benchmark: {}'.format(benchmark))
                sys.exit(1)

    @staticmethod
    def prepare_benchmark(benchmark,
                          benchmarks_directory):
        """We need to store all benchmarks in the same location.

        Parameters
        ----------
        benchmark : str

        benchmark_directory: str
        """
        Inst2Vec.__data_directory_ir = tempfile.mkdtemp(suffix='_ir')

        index = benchmark.find('.')
        src = os.path.join(
            benchmarks_directory,
            benchmark[:index],
            benchmark[index+1:]
        )
        dst = os.path.join(Inst2Vec.__data_directory_ir, benchmark)

        cmdline = 'ln -s {} {}'.format(src, dst)

        # Invoke the disassembler
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.error('Prepare benchmark: {}'.format(benchmark))
            sys.exit(1)

    @classproperty
    def data_directory_ir(cls):
        """Getter."""
        return cls.__data_directory_ir

    @classproperty
    def data_directory_seq(cls):
        """Getter."""
        return cls.__data_directory_seq

    @staticmethod
    def remove_data_directory():
        """Remove data directory."""
        data_dir = Inst2Vec.__data_directory_ir[:-3]
        cmdline = 'rm -rf {}*'.format(data_dir)
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.warning('Removing data directory')

    @staticmethod
    def compile_and_extract(benchmarks_filename,
                            benchmarks_directory,
                            sequence,
                            embeddings_file=None,
                            vocabulary_dir=None):
        """Compile the benchmarks and extract int2vec representation.

        Parameters
        ----------
        benchmarks_filename : str

        benchmarks_directory: str

        sequence : list

        embedding_file : str

        vocabulary_dir : str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        # Verify what data to use
        if not embeddings_file:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.sys.exit(1)
            embeddings_file = os.path.join(top_dir,
                                           Inst2Vec.__data_dir,
                                           Inst2Vec.__embeddings_file)

        if not vocabulary_dir:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.sys.exit(1)
            vocabulary_dir = os.path.join(top_dir, Inst2Vec.__data_dir)

        # Prepare the directories
        Inst2Vec.prepare_benchmarks(benchmarks_filename,
                                    benchmarks_directory)

        for benchmark in os.listdir(Inst2Vec.__data_directory_ir):
            bench_dir = os.path.join(Inst2Vec.__data_directory_ir, benchmark)
            Engine.compile_(bench_dir,
                            'opt',
                            Sequence.name_pass_to_string(sequence))
            Engine.disassemble(bench_dir, 'a.out_o')

        # IR -> embeddings' index
        Inst2Vec.__data_directory_seq = llvm_ir_to_trainable(
            Inst2Vec.__data_directory_ir,
            vocabulary_dir
        )

        # Cleanup the directories
        for benchmark in os.listdir(Inst2Vec.__data_directory_ir):
            bench_dir = os.path.join(Inst2Vec.__data_directory_ir, benchmark)
            Engine.cleanup(bench_dir, 'opt')

        # Load the embeddings
        embeddings = IO.load_pickle_or_fail(embeddings_file)

        # Embeddings' index -> embeedings
        processed = {}
        for directory in os.listdir(Inst2Vec.__data_directory_seq):
            bench_dir = os.path.join(Inst2Vec.__data_directory_seq, directory)
            file_ = open('{}/a.out_o.csv'.format(bench_dir), 'r')
            reader = csv.reader(file_)
            data = [embeddings[int(row[0])] for row in reader]
            processed[directory] = np.matrix(data)
            file_.close()

        # Return the representation
        return processed

    @staticmethod
    def extract(embeddings_file=None,
                vocabulary_dir=None):
        """Extract int2vec representation.

        This method have to use for processing only 1 benchmark.

        Parameters
        ----------
        embeddings_file : str

        vocabulary_dir : str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        # Verify what data to use
        if not embeddings_file:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.sys.exit(1)
            embeddings_file = os.path.join(top_dir,
                                           Inst2Vec.__data_dir,
                                           Inst2Vec.__embeddings_file)

        if not vocabulary_dir:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.sys.exit(1)
            vocabulary_dir = os.path.join(top_dir, Inst2Vec.__data_dir)

        Inst2Vec.__data_directory_seq = llvm_ir_to_trainable(
            Inst2Vec.__data_directory_ir,
            vocabulary_dir
        )

        # Load the embeddings
        embeddings = IO.load_pickle_or_fail(embeddings_file)

        # Embeddings' index -> embeedings
        directory = os.listdir(Inst2Vec.__data_directory_seq)
        if len(directory) > 1:
            lg.error('Extract founded more than 1 directory.')
            sys.exit(1)

        directory = os.path.join(Inst2Vec.__data_directory_seq, directory[0])
        benchmarks = glob.glob('{}/*.csv'.format(directory))

        # Embeddings' index -> embeedings
        processed = {}
        for benchmark in benchmarks:
            file_ = open(benchmark, 'r')
            reader = csv.reader(file_)
            data = [embeddings[int(row[0])] for row in reader]
            name = benchmark.replace('{}/'.format(directory), '')
            name = name.replace('_seq.csv', '')
            processed[name] = np.matrix(data)
            file_.close()

        return processed

    @staticmethod
    def get_unknown_embeddings():
        """Get unknown embeddings."""
        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.sys.exit(1)
        embeddings_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__embeddings_file)

        # Load the embeddings
        embeddings = IO.load_pickle_or_fail(embeddings_file)
        # Return embeddings
        return embeddings[8564]
