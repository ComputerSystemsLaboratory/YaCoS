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
import yaml

from absl import logging as lg

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence

from yacos.info.image import bit2vec

class Prog2Image:
    """Prog2Image Representation."""

    __version__ = '2.0.0'

    @staticmethod
    def compile_and_extract(benchmarks_base_directory,
                            benchmarks_filename,
                            sequence='-O0',
                            columns=256):
        """Compile the benchmark and extract prog2image representation.
            The benchmark directory must have a Makefile.opt that generates
            the bytecode as a.out.bc (or a.out_o.bc)
        Parameters
        ----------
        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

        ret = []
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)

        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            Engine.compile(benchmark_dir,'opt',sequence)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_array = bit2vec.get_array(bytecode_file)
            benchmark_emb = bit2vec(benchmark_array,columns=columns)
            ret[benchmark] = benchmark_emb.create_npz()
        
        return ret

    @staticmethod
    def extract(benchmarks_filename):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

class LBPeq:
    """LBPeq Representation."""

    __version__ = '2.0.0'

    @staticmethod
    def compile_and_extract(benchmarks_filename,
                            sequence):
        """Compile the benchmark and extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

    @staticmethod
    def extract(benchmarks_filename):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

class LBPif:
    """LBPeq Representation."""

    __version__ = '2.0.0'

    @staticmethod
    def compile_and_extract(benchmarks_filename,
                            sequence):
        """Compile the benchmark and extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

    @staticmethod
    def extract(benchmarks_filename):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
