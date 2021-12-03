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
import subprocess
import glob
from dataclasses import dataclass
from absl import logging as lg

from yacos.essential import Engine
from yacos.essential import Sequence


class BenchmarkReduction:
    """Create a small benchmark, based on number of LLVM instructions."""

    __version__ = '1.0.0'

    __flags = None

    @dataclass
    class Flags:
        """Reduce flags.

        goals : dict

        baseline : str
            The compiler optimization level: Oz

        compiler : str
            The compiler to use.

        benchmarks_directory : str

        output_directory: str
            The new directory
        """

        goals: dict
        baseline: str
        compiler: str
        benchmarks_directory: str
        output_directory: str

    def __init__(self,
                 baseline,
                 benchmarks_directory,
                 output_directory):
        """Initialize a BenchmarkReduction object.

        Parameters
        ----------
        baseline : str
            Example: 'Oz'

        benchmarks_directory : str

        output_directory : str
        """
        self.__flags = self.Flags({'llvm_instructions': 1.0},
                                  '-{}'.format(baseline),
                                  'opt',
                                  benchmarks_directory,
                                  output_directory)

    def run(self,
            benchmark,
            sequence):
        """Invoke C-Reduce to create a small benchmark.

        Parameter
        ----------
        benchmark : str

        sequence : list

        Return
        ------
        ret : bool
            True if reduction ok, otherwise False.
        """
        index = benchmark.find('.')
        bench_in_dir = os.path.join(self.__flags.benchmarks_directory,
                                    benchmark[:index],
                                    benchmark[index+1:])

        if not os.path.isdir(bench_in_dir):
            lg.error('The directory {} does not exist.'.format(bench_in_dir))
            sys.exit(1)

        # 1. Calculate the difference (gain)

        # 1.1. Baseline's goal value
        baseline_goal_value = Engine.evaluate(self.__flags.goals,
                                              self.__flags.baseline,
                                              self.__flags.compiler,
                                              bench_in_dir)

        # 1.2. Sequence's goal value
        sequence_goal_value = Engine.evaluate(self.__flags.goals,
                                              Sequence.name_pass_to_string(
                                                  sequence
                                              ),
                                              self.__flags.compiler,
                                              bench_in_dir)

        diff = int(baseline_goal_value - sequence_goal_value)
        if diff <= 0:
            lg.info(
                'Warning: It is not possible to reduce the code ({}).'.format(
                    diff
                )
            )
            return False

        # 2. Prepare the benchmark
        bench_out_dir = os.path.join(self.__flags.output_directory,
                                     benchmark[:index],
                                     benchmark[index+1:])

        os.makedirs(bench_out_dir, exist_ok=True)

        cmdline = 'cp {0}/*.c {1}'.format(bench_in_dir,
                                          bench_out_dir)

        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.fatal('Prepare the benchmark')

        # 3. Create the C-Reduce script
        filename = '{}/test.sh'.format(bench_out_dir)

        fout = open(filename, 'w')
        fout.write('#!/bin/bash\n')
        fout.write('DIFF={}\n'.format(diff))
        fout.write('LIB=libMilepostStaticFeatures.so\n')
        fout.write('PASSES="{}"\n'.format(
            Sequence.name_pass_to_string(sequence))
        )
        fout.write('clang -Xclang -disable-O0-optnone -w -c -emit-llvm *.c\n')
        fout.write('llvm-link *.bc -S -o creduce.bc\n')
        fout.write('opt --disable-output -load $LIB {} \
        --msf creduce.ll 2> msf.txt\n'.format(self.__flags.baseline))
        fout.write('size_Oz=`grep f25 msf.txt | \
        awk \'{total += $NF} END { print total }\'`\n')
        fout.write('rm -f msf.txt\n')
        fout.write('opt --disable-output -load $LIB \
        $PASSES --msf creduce.bc 2> msf.txt\n')
        fout.write('size_PASSES=`grep f25 msf.txt | \
        awk \'{total += $NF} END { print total }\'`\n')
        fout.write('rm -f msf.txt\n')
        fout.write('diff=$(($size_Oz-$size_PASSES))\n')
        fout.write('if [[ $diff -eq $DIFF ]]; then\n')
        fout.write('   exit 0\n')
        fout.write('else\n')
        fout.write('   exit 1\n')
        fout.write('fi\n')
        fout.close()

        cmdline = 'chmod +x {}/test.sh'.format(bench_out_dir)

        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.fatal('Writing C-Reduce script')

        # 4. Invoke C-Reduce
        prefix = '{}/'.format(bench_out_dir)
        sources = glob.glob('{}*.c'.format(prefix))
        sources = [source.replace(prefix, '').replace('.c', '')
                   for source in sources]
        sources = ''.join(sources)

        cmdline = 'cd {0} ; ' \
                  + 'creduce ./test.sh {1}.c > creduce.log 2> creduce.log'
        cmdline.format(bench_out_dir, sources)

        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.fatal('Invoke C-Reduce')

        # 5. Create the new benchmark directory
        cmdline = 'cp {0}/Makefile.* {0}/compile.sh {1}'.format(bench_in_dir,
                                                                bench_out_dir)

        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.fatal('Invoke C-Reduce - Move the small file')

        return True
