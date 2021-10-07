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
# Calculate MCoeff.
#

import os
import sys

from tqdm import tqdm
from absl import app, flags, logging

from yacos.essential import IO
from yacos.essential import Similarity


def execute(argv):
    """Find the euclidean distance from test to training data."""
    del argv

    # Training benchmarks
    training_benchmarks = IO.load_yaml_or_fail(FLAGS.training_benchs_filename)
    if not training_benchmarks:
        logging.error('There are no training benchmarks to process')
        sys.exit(1)

    # Test benchmarks
    test_benchmarks = IO.load_yaml_or_fail(FLAGS.test_benchs_filename)
    if not test_benchmarks:
        logging.error('There are no test benchmarks to process')
        sys.exit(1)

    # Process each benchmark.
    for test_benchmark in tqdm(test_benchmarks, desc='Processing'):
        test_index = test_benchmark.find('.')
        test_suite = test_benchmark[:test_index]
        test_bench = test_benchmark[test_index+1:]

        results = {}
        filename = os.path.join(FLAGS.test_data_directory,
                                test_suite,
                                '{}.yaml'.format(test_bench))

        test_data = IO.load_yaml_or_fail(filename)

        # Create the results directory
        results_dir = os.path.join(FLAGS.results_directory,
                                   test_suite)
        os.makedirs(results_dir, exist_ok=True)

        for training_benchmark in training_benchmarks:
            training_index = training_benchmark.find('.')
            training_suite = training_benchmark[:training_index]
            training_bench = training_benchmark[training_index+1:]

            filename = os.path.join(FLAGS.training_data_directory,
                                    training_suite,
                                    '{}.yaml'.format(training_bench))

            training_data = IO.load_yaml_or_fail(filename)

            results[training_bench] = Similarity.mcoeff(
                                        test_data,
                                        training_data
                                      )

        filename = os.path.join(results_dir,
                                '{}.yaml'.format(test_bench))
        IO.dump_yaml(results, filename)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('training_benchs_filename',
                        None,
                        'Training benchmarks')
    flags.DEFINE_string('test_benchs_filename',
                        None,
                        'Test benchmark')
    flags.DEFINE_string('training_data_directory',
                        None,
                        'Training data directory')
    flags.DEFINE_string('test_data_directory',
                        None,
                        'Test data directory')

    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')

    flags.mark_flag_as_required('training_benchs_filename')
    flags.mark_flag_as_required('test_benchs_filename')
    flags.mark_flag_as_required('training_data_directory')
    flags.mark_flag_as_required('test_data_directory')
    flags.mark_flag_as_required('results_directory')

    FLAGS = flags.FLAGS

    app.run(execute)
