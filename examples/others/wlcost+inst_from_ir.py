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
# Extract Wu-Larus costs from each functions, and also the
# number of llvm instructions.
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Engine
from yacos.essentials import Sequence
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def execute(argv):
    """Evaluate N sequences."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # The sequences.
    sequences = IO.load_yaml_or_fail(FLAGS.sequences_filename)
    if not sequences:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Instantiate the LLVM driver.
    driver = LLVMDriver()
    # Instantiate the builder.
    builderC = R.LLVMWLCostBuilder(driver)
    builderI = R.LLVMInstsBuilder(driver)

    # Process each benchmark.
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        bench_dir = os.path.join(FLAGS.benchmarks_directory,
                                 suite,
                                 bench)

        if not os.path.isdir(bench_dir):
            continue

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite)

        # Create the output directory.
        os.makedirs(results_dir, exist_ok=True)

        # Verify report.
        if FLAGS.suffix:
            filename = '{}/{}_{}.yaml'.format(
                results_dir,
                bench,
                FLAGS.suffix
            )
        else:
            filename = '{}/{}.yaml'.format(results_dir, bench)

        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        results = {}
        for key, data in sequences.items():

            # Compile de benchmark
            Engine.compile_(bench_dir,
                            'opt',
                            Sequence.name_pass_to_string(data['seq']))

            # Read the source files.
            source = '{}/a.out_o.bc'.format(bench_dir)

            if not os.path.isfile(source):
                continue

            # Extract the number of llvm instructions.
            instructions = builderI.ir_to_info(source)
            # Extract the costs.
            costs = builderC.ir_to_info(source)

            instructions_ = {}
            for info in instructions.functionInfos:
                instructions_[info.name] = info.instructions

            costs_ = {}
            for info in costs.functionInfos:
                costs_[info.name] = {'recipThroughput': info.recipThroughput,
                                     'latency': info.latency,
                                     'codeSize': info.codeSize}

            results[key] = {'seq': data['seq'],
                            'instructions': instructions_,
                            'costs': costs_}

            Engine.cleanup(bench_dir, 'opt')

        # Store the results.
        IO.dump_yaml(results, filename)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences')
    flags.DEFINE_string('suffix',
                        '',
                        'Filename suffix')

    flags.mark_flag_as_required('sequences_filename')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
