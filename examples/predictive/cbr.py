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
# Predictive compilation using a case-based reasoning strategy.
# Compile the whole program using a specific sequence, take from the most
# similar training program.
#
# This version does not extract features.
#
# This version use:
#    - yacos.algorithm.CBR
#    - Milepost Static Features
#

import os
import sys
import numpy as np

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.algorithm import CBR


def sequences(training_directory,
              training_benchmarks):
    """Extract training sequences."""
    data = {}
    for benchmark in training_benchmarks:
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        filename = '{}/{}/{}.yaml'.format(training_directory, suite, bench)
        data[bench] = IO.load_yaml_or_fail(filename)

    return data


def similarity(similarity_directory,
               training_benchmarks,
               test_benchmark):
    """Similarity"""
    filename = '{}/{}.npz'.format(similarity_directory, test_benchmark)
    data = np.load(filename, allow_pickle=True)

    training_similarity = {}
    for benchmark in training_benchmarks:
        index = benchmark.find('.')
        bench = benchmark[index+1:]

        index = list(data['training']).index(bench)
        training_similarity[bench] = data['values'][index]

    return training_similarity


def baseline_goal_value(training_benchmarks_filename,
                        levels_directory,
                        compiler,
                        level):
    """Extract the goal value."""
    # The test benchmarks
    benchmarks = IO.load_yaml_or_fail(training_benchmarks_filename)
    if not benchmarks:
        logging.error('There are no training benchmarks to process')
        sys.exit(1)

    data = {}
    for benchmark in benchmarks:
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        filename = '{}/{}/{}.yaml'.format(levels_directory, suite, bench)
        levels = IO.load_yaml_or_fail(filename)
        data[bench] = levels[compiler][level]['goal']

    return data


def execute(argv):
    """Predictive compilation."""
    del argv

    FLAGS = flags.FLAGS
    # The test benchmarks

    training_benchmarks = IO.load_yaml_or_fail(
                            FLAGS.training_benchmarks_filename
                          )
    if not training_benchmarks:
        logging.error('There are no training benchmarks to process')
        sys.exit(1)

    # The test benchmarks
    test_benchmarks = IO.load_yaml_or_fail(FLAGS.test_benchmarks_filename)
    if not test_benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Initialize a CBR object
    cbr = CBR(Goal.prepare_goal(FLAGS.goals, FLAGS.weights),
              'opt',
              FLAGS.test_benchmarks_directory,
              FLAGS.working_set,
              FLAGS.times,
              FLAGS.tool,
              FLAGS.verify_output)

    # Process each benchmark
    for benchmark in tqdm(test_benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        bench_dir = os.path.join(FLAGS.test_benchmarks_directory,
                                 suite,
                                 bench)

        if not os.path.isdir(bench_dir):
            continue

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite)

        # Create the results directory for the suite
        os.makedirs(results_dir, exist_ok=True)

        for selection in ['elite', 'just', 'nearly']:
            # Verify report
            filename = '{}/{}_{}_{}.yaml'.format(results_dir,
                                                 bench,
                                                 selection,
                                                 FLAGS.nof_sequences)

            if FLAGS.verify_report and os.path.isfile(filename):
                continue

            training_sequences = sequences(FLAGS.training_directory,
                                           training_benchmarks)

            training_baseline_goal_value = baseline_goal_value(
                    FLAGS.training_benchmarks_filename,
                    FLAGS.levels_directory,
                    FLAGS.compiler,
                    FLAGS.level
            )

            training_similarity = similarity(FLAGS.similarity_directory,
                                             training_benchmarks,
                                             bench)

            # Run CBR
            cbr.run(benchmark,
                    training_sequences,
                    training_baseline_goal_value,
                    training_similarity,
                    selection,
                    FLAGS.nof_sequences)

            if cbr.results:
                # Store the results
                IO.dump_yaml(cbr.results, filename)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('training_benchmarks_filename',
                        None,
                        'Training benchmarks')
    flags.DEFINE_string('test_benchmarks_filename',
                        None,
                        'Test benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('levels_directory',
                        None,
                        'Levels directory')
    flags.DEFINE_string('similarity_directory',
                        None,
                        'Similarity directory')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')
    flags.DEFINE_list('goals',
                      None,
                      'Goal')
    flags.DEFINE_list('weights',
                      None,
                      'Weights')
    flags.DEFINE_string('test_benchmarks_directory',
                        None,
                        'Test benchmarks directory')
    flags.DEFINE_string('training_directory',
                        None,
                        'Training directory')
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
                         'The goal is valid if the ouput is correct')
    flags.DEFINE_enum('compiler',
                      'opt',
                      ['clang', 'opt'],
                      'Compiler')
    flags.DEFINE_enum('level',
                      'Oz',
                      ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz'],
                      'Compiler optimization level')
    flags.DEFINE_integer('nof_sequences',
                         10,
                         'Number of sequences',
                         lower_bound=1)

    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('training_benchmarks_filename')
    flags.mark_flag_as_required('test_benchmarks_filename')
    flags.mark_flag_as_required('test_benchmarks_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('levels_directory')
    flags.mark_flag_as_required('similarity_directory')
    flags.mark_flag_as_required('training_directory')

    app.run(execute)
