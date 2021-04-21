#! /usr/bin/env python3

"""
Copyright 2021 Anderson Faustino da Silva.

Licensed under the Apache License, Version 2.0 (the "License");
you m
ay not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#
# Find the best k sequences
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.algorithms import BestK


def execute(argv):
    """Find the best K sequences, from training data."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.fatal('There are no training benchmarks to process.')

    # Create the output directory.
    os.makedirs(FLAGS.results_directory, exist_ok=True)

    # Verify directories.
    if not os.path.isdir(FLAGS.training_directory):
        logging.error('Training directory {} does not exit.'.format(
            FLAGS.training_directory)
        )
        sys.exit(1)

    if not os.path.isdir(FLAGS.baseline_directory):
        logging.error('Baseline directory {} does not exit.'.format(
            FLAGS.baseline_directory)
        )
        sys.exit(1)

    # Initialize a BestK object.
    bestk = BestK(FLAGS.training_directory,
                  FLAGS.baseline_directory)

    for k in tqdm(FLAGS.k, desc='Best-k'):
        filename = '{}/best_{}.yaml'.format(FLAGS.results_directory, k)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        # Run the algorithm.
        bestk.run(benchmarks,
                  FLAGS.compiler,
                  FLAGS.baseline,
                  int(k))

        # Store the results
        IO.dump_yaml(bestk.results, filename)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('training_directory',
                        None,
                        'Training directory')
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Training benchmarks')
    flags.DEFINE_enum('baseline',
                      'Oz',
                      ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz'],
                      'Baseline')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')
    flags.DEFINE_list('k',
                      None,
                      'Number of sequences: [k1,k2,...]')
    flags.DEFINE_string('baseline_directory',
                        None,
                        'Compiler optimization levels directory')
    flags.DEFINE_enum('compiler',
                      'clang',
                      ['clang', 'opt'],
                      'Compiler')

    flags.mark_flag_as_required('k')
    flags.mark_flag_as_required('baseline_directory')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('training_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
