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
# Improved Batch Elimination
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.algorithm import ImprovedBatchElimination


def execute(argv):
    """Execute Improved Batch Elimination."""
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

    # Initialize a ImprovedBatchElimination object.
    ibe = ImprovedBatchElimination(FLAGS.points,
                                   Goal.prepare_goal(FLAGS.goals,
                                                       FLAGS.weights),
                                   'opt',
                                   FLAGS.benchmarks_directory,
                                   FLAGS.working_set,
                                   FLAGS.times,
                                   FLAGS.tool,
                                   FLAGS.verify_output)

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

        # Run the algorithm.
        ibe.run(benchmark, sequences)

        if ibe.results:
            # Store the results
            IO.dump_yaml(ibe.results, filename)


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
    flags.DEFINE_list('goals',
                      None,
                      'Goal')
    flags.DEFINE_list('weights',
                      None,
                      'Weights')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_integer('working_set',
                         0,
                         'Working set',
                         lower_bound=0)
    flags.DEFINE_integer('times',
                         3,
                         'Execution/compile times',
                         lower_bound=3)
    flags.DEFINE_enum('tool',
                      'perf',
                      ['perf', 'hyperfine'],
                      'Execution tool')
    flags.DEFINE_boolean('verify_output',
                         False,
                         'The goal is only valid if the ouput is correct')
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences')
    flags.DEFINE_integer('points',
                         1,
                         'The number of points',
                         lower_bound=1)
    flags.DEFINE_string('suffix',
                        '',
                        'Filename suffix')

    flags.mark_flag_as_required('sequences_filename')
    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
