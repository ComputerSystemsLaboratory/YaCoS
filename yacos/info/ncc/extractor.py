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

from absl import logging as lg

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.info.ncc import llvm_ir_to_trainable


class Inst2Vec:
    """Inst2Vec Representation."""

    class classproperty(property):
        """classproperty decorator."""

        def __get__(self, cls, owner):
            """Decorate."""
            return classmethod(self.fget).__get__(None, owner)()

    __version__ = '2.0.0'

    __data_directory_ir = None
    __data_directory_seq = None

    __data_dir = 'yacos/data/inst2vec'
    __embeddings_file = 'inst2vec_augmented_embeddings.pickle'
    __vocabulary_file = 'inst2vec_augmented_dictionary.pickle'

    @classproperty
    def data_directory_ir(cls):
        """Getter."""
        return cls.__data_directory_ir

    @classproperty
    def data_directory_seq(cls):
        """Getter."""
        return cls.__data_directory_seq

    @classproperty
    def unknown(cls):
        """Get unknown index and embeddings."""
        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        # Load the vocabulary
        vocabulary_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__vocabulary_file)
        vocabulary = IO.load_pickle_or_fail(vocabulary_file)

        # Load the embeddings
        embeddings_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__embeddings_file)
        embeddings = IO.load_pickle_or_fail(embeddings_file)

        unk_index = vocabulary['!UNK']
        unk_embeddings = embeddings[unk_index]

        del vocabulary
        del embeddings

        # Return embeddings
        return unk_index, unk_embeddings

    @classproperty
    def length(cls):
        """Get embeddings length."""
        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        # Load the vocabulary
        vocabulary_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__vocabulary_file)
        vocabulary = IO.load_pickle_or_fail(vocabulary_file)

        # Load the embeddings
        embeddings_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__embeddings_file)
        embeddings = IO.load_pickle_or_fail(embeddings_file)

        unk_index = vocabulary['!UNK']
        unk_embeddings = embeddings[unk_index]

        del vocabulary
        del embeddings

        # Return embeddings
        return len(unk_embeddings)

    @classproperty
    def vocabulary(cls):
        """Get the vocabulary."""
        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        # Load the vocabulary
        vocabulary_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__vocabulary_file)
        vocabulary = IO.load_pickle_or_fail(vocabulary_file)

        # Return the vocabulary
        return vocabulary

    @classproperty
    def embeddings(cls):
        """Get the embeddings."""
        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        # Load the embeddings
        embeddings_file = os.path.join(top_dir,
                                       Inst2Vec.__data_dir,
                                       Inst2Vec.__embeddings_file)
        embeddings = IO.load_pickle_or_fail(embeddings_file)

        # Return the embeddings
        return embeddings

    @staticmethod
    def prepare_benchmarks(benchmarks_directory,
                           benchmarks_filename):
        """We need to store all benchmarks in the same location.

        Parameters
        ----------
        benchmarks_directory : str

        benchmarks_filename: str
        """
        if not os.path.isabs(benchmarks_directory):
            lg.error('Benchmarks directory is not an absolute directory')
            sys.exit(1)

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
    def prepare_benchmark(benchmark_directory,
                          benchmark=''):
        """We need to store all benchmarks in the same location.

        Parameters
        ----------
        benchmark_directory : str

        benchmark: str
        """
        if not os.path.isabs(benchmark_directory):
            lg.error('Benchmark directory is not an absolute directory')
            sys.exit(1)

        Inst2Vec.__data_directory_ir = tempfile.mkdtemp(suffix='_ir')

        index = benchmark.find('.')
        src = os.path.join(
                    benchmark_directory,
                    benchmark[:index],
                    benchmark[index+1:]
              ) if benchmark else benchmark_directory

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
    def compile_and_extract(benchmarks_directory,
                            benchmarks_filename,
                            sequence,
                            embeddings_file=None,
                            vocabulary_dir=None,
                            data_type='index_embeddings'):
        """Compile the benchmarks and extract int2vec representation.

        Parameters
        ----------
        benchmarks_directory : str

        benchmarks_filename: str

        sequence : list

        embedding_file : str

        vocabulary_dir : str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        if not os.path.isabs(benchmarks_directory):
            lg.error('Benchmarks directory is not an absolute directory')
            sys.exit(1)

        # Verify what data to use
        if not embeddings_file:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.exit(1)
            embeddings_file = os.path.join(top_dir,
                                           Inst2Vec.__data_dir,
                                           Inst2Vec.__embeddings_file)

        if not vocabulary_dir:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.exit(1)
            vocabulary_dir = os.path.join(top_dir, Inst2Vec.__data_dir)

        # Prepare the directories
        Inst2Vec.prepare_benchmarks(benchmarks_directory,
                                    benchmarks_filename)

        for benchmark in os.listdir(Inst2Vec.__data_directory_ir):
            bench_dir = os.path.join(Inst2Vec.__data_directory_ir, benchmark)
            Engine.compile(bench_dir,
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

            index = []
            emb = []
            for row in reader:
                index.append(int(row[0]))
                emb.append(list(embeddings[int(row[0])]))

            if data_type == 'index':
                processed[directory] = index
            elif data_type == 'embeddings':
                processed[directory] = emb
            elif data_type == 'index_embeddings':
                processed[directory] = (index, emb)
            else:
                lg.error('Data type {} does not exist.'.format(data_type))
                sys.exit(1)

            file_.close()

        # Free embeddings
        del embeddings

        # Return the representation
        return processed

    @staticmethod
    def extract(embeddings_file=None,
                vocabulary_dir=None,
                data_type='index_embeddings'):
        """Extract int2vec representation.

        Process only 1 benchmark.

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
                sys.exit(1)
            embeddings_file = os.path.join(top_dir,
                                           Inst2Vec.__data_dir,
                                           Inst2Vec.__embeddings_file)

        if not vocabulary_dir:
            top_dir = os.path.join(os.environ.get('HOME'), '.local')
            if not os.path.isdir(os.path.join(top_dir, 'yacos')):
                lg.error('YaCoS data does not exist.')
                sys.exit(1)
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

            index = []
            emb = []
            for row in reader:
                index.append(int(row[0]))
                emb += list(embeddings[int(row[0])])

            name = benchmark.replace('{}/'.format(directory), '')
            name = name.replace('_seq.csv', '')

            if data_type == 'index':
                processed[name] = index
            elif data_type == 'embeddings':
                processed[name] = emb
            elif data_type == 'index_embeddings':
                processed[name] = (index, emb)
            else:
                lg.error('Data type {} does not exist.'.format(data_type))
                sys.exit(1)

            file_.close()

        # Free embeddings
        del embeddings

        # Return the representation
        return processed
