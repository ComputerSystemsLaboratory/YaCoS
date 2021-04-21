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
# Measure the distance between two numeric vectors.
# This code finds similar programs based on numeric features.
#

import os
import sys
import numpy as np

from absl import app, flags, logging

from yacos.essentials import IO
from yacos.essentials import Similarity


def prepare_data(representation_directory,
                 benchmarks):
    """Load the data into a dictonary."""
    representations = {}
    for benchmark in benchmarks:
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(representation_directory,
                                     suite_name)

        filename = '{}/{}.npz'.format(benchmark_dir, bench_name)
        if not os.path.isfile(filename):
            logging.error('File does not exist: {}.'.format(filename))
            continue

        data = np.load(filename)
        if data:
            representations[benchmark] = [sum(x) for x in zip(*data['values'])]

    return representations


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

    # Verify training directory
    if not os.path.isdir(FLAGS.training_representation_directory):
        logging.error('Training directory {} does not exist.'.format(
            FLAGS.training_representation_directory))
        sys.exit(1)

    # Verify test directory
    if not os.path.isdir(FLAGS.test_representation_directory):
        logging.error('Test directory {} does not exist.'.format(
            FLAGS.test_representation_directory))
        sys.exit(1)

    training_data = prepare_data(FLAGS.training_representation_directory,
                                 training_benchmarks)
    test_data = prepare_data(FLAGS.test_representation_directory,
                             test_benchmarks)

    """ Measure the distance."""
    if FLAGS.distance == 'euclidean':
        training_index, test_index, distance = Similarity.euclidean_distance_from_data(training_data, test_data)
    elif FLAGS.distance == 'manhattan':
        training_index, test_index, distance = Similarity.manhattan_distance_from_data(training_data, test_data)
    elif FLAGS.distance == 'cosine':
        training_index, test_index, distance = Similarity.cosine_distance_from_data(training_data, test_data)

    # Store the distance
    for i, test_bench in enumerate(test_index):
        index = test_bench.find('.')
        test_suite_name = test_bench[:index]
        test_bench_name = test_bench[index+1:]

        results = {}
        for j, training_bench in enumerate(training_index):
            index = training_bench.find('.')
            training_suite_name = training_bench[:index]
            training_bench_name = training_bench[index+1:]

            if training_suite_name not in results:
                results[training_suite_name] = {}

            results[training_suite_name][
                                            training_bench_name
                                        ] = float(distance[i][j])

        for training_suite_name, training_distance in results.items():
            results_dir = os.path.join(FLAGS.results_directory,
                                       test_suite_name,
                                       training_suite_name)

            # Create the results directory
            os.makedirs(results_dir, exist_ok=True)

            training = list(training_distance.keys())
            values = list(training_distance.values())

            filename = '{}/{}'.format(results_dir, test_bench_name)
            np.savez_compressed(filename, training=training, values=values)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('training_benchs_filename',
                        None,
                        'Training benchmarks')
    flags.DEFINE_string('test_benchs_filename',
                        None,
                        'Test benchmark')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('training_representation_directory',
                        None,
                        'Training representation directory')
    flags.DEFINE_string('test_representation_directory',
                        None,
                        'Test representation directory')
    flags.DEFINE_enum('distance',
                      'euclidean',
                      ['euclidean', 'manhattan', 'cosine'],
                      "Distance")

    flags.mark_flag_as_required('training_representation_directory')
    flags.mark_flag_as_required('training_benchs_filename')
    flags.mark_flag_as_required('test_representation_directory')
    flags.mark_flag_as_required('test_benchs_filename')
    flags.mark_flag_as_required('results_directory')

    FLAGS = flags.FLAGS

    app.run(execute)
