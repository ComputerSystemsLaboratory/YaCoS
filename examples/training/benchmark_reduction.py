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
# Benchmark reduction
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Sequence
from yacos.algorithms import BenchmarkReduction


def execute(argv):
    """Create a small sequence."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.fatal('There are no benchmarks to process')

    # Verify directory.
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    if not os.path.isdir(FLAGS.training_directory):
        logging.error('Training directory {} does not exist.'.format(
            FLAGS.training_directory)
        )
        sys.exit(1)

    # Create the output directory.
    os.makedirs(FLAGS.results_directory, exist_ok=True)

    # Initialize a BenchmarkReduction object.
    bred = BenchmarkReduction(FLAGS.baseline,
                              FLAGS.benchmarks_directory,
                              FLAGS.results_directory)

    # Process each benchmark.
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        training_dir = os.path.join(FLAGS.training_directory,
                                    suite)

        sequences = IO.load_yaml_or_fail('{}/{}.yaml'.format(training_dir,
                                                             bench))
        sequence = Sequence.get_the_best(sequences)
        for _, seq_data in sequence.items():
            sequence = seq_data['seq']

        # Run the algorithm.
        bred.run(benchmark, sequence)


# Execute
if __name__ == '__main__':
    flags.DEFINE_enum('baseline',
                      'Os',
                      ['O0', 'Os', 'Oz'],
                      'Baseline')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    # app
    flags.DEFINE_string('training_directory',
                        None,
                        'Training data directory')
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')

    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('training_directory')

    app.run(execute)
