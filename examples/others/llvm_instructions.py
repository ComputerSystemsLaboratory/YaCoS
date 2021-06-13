#! /usr/bin/env python3

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

#
# Extract functions name from a program.
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Engine
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def process(benchmark_dir):
    """Transform Source code into representation."""
    # Instantiate the Clang Driver
    driver = LLVMDriver()

    # Instantiate the builder
    builder = R.LLVMInstsBuilder(driver)

    # Generate target code
    Engine.compile(benchmark_dir, 'opt', '-O0')

    # Process
    source = '{}/a.out_o.bc'.format(benchmark_dir)
    extractionInfo = builder.ir_to_info(source)
    insts = [data.instructions for data in extractionInfo.functionInfos]

    return sum(insts)


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Process each benchmark
    info = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                     suite_name,
                                     bench_name)

        if not os.path.isdir(benchmark_dir):
            continue

        info[bench_name] = process(benchmark_dir)

    # Create the output directory.
    os.makedirs(FLAGS.results_directory, exist_ok=True)

    # Store the results
    filename = '{}/{}.yaml'.format(FLAGS.results_directory,
                                   FLAGS.results_filename)
    IO.dump_yaml(info, filename)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('results_filename',
                        None,
                        'Results filename')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('results_filename')

    app.run(execute)
