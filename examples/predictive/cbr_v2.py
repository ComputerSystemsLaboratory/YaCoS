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
# This version does not use yacos.algorithm.CBR and does not extract features.
#

import os
import sys
import numpy as np

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Goals
from yacos.essentials import Engine
from yacos.essentials import Sequence


def filter_sequences(training_directory,
                     training_suite,
                     training_bench,
                     levels_directory,
                     compiler,
                     level,
                     sequences):
    """Filter training sequences."""
    # Load training sequences
    filename = '{}/{}/{}.yaml'.format(training_directory,
                                      training_suite,
                                      training_bench)
    training_sequences = IO.load_yaml_or_fail(filename)

    # Load the compiler optimization levels
    filename = '{}/{}/{}.yaml'.format(levels_directory,
                                      training_suite,
                                      training_bench)
    levels = IO.load_yaml_or_fail(filename)
    level_goal = levels[compiler][level]['goal']

    good_sequences = {}
    for key, seq_data in training_sequences.items():
        if seq_data['goal'] < level_goal:
            good_sequences[key] = seq_data

    return good_sequences


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

    # Process each benchmark
    for test_benchmark in tqdm(test_benchmarks, desc='Processing'):
        test_index = test_benchmark.find('.')
        test_suite = test_benchmark[:test_index]
        test_bench = test_benchmark[test_index+1:]

        bench_dir = os.path.join(FLAGS.test_benchmarks_directory,
                                 test_suite,
                                 test_bench)

        if not os.path.isdir(bench_dir):
            continue

        results_dir = os.path.join(FLAGS.results_directory,
                                   test_suite)

        # Create the results directory for the suite
        os.makedirs(results_dir, exist_ok=True)

        # Verify report
        results_filename = '{}/{}_{}_{}.yaml'.format(results_dir,
                                                     test_bench,
                                                     FLAGS.selection,
                                                     FLAGS.nof_sequences)

        if FLAGS.verify_report and os.path.isfile(results_filename):
            continue

        # Get similarity
        filename = '{}/{}.npz'.format(FLAGS.similarity_directory, test_bench)
        similarity = np.load(filename, allow_pickle=True)

        # Rank training programs
        training_similarity = []
        for training_benchmark in training_benchmarks:
            tr_index = training_benchmark.find('.')
            tr_suite = training_benchmark[:tr_index]
            tr_bench = training_benchmark[tr_index+1:]

            index = list(similarity['training']).index(tr_bench)
            training_similarity.append((similarity['values'][index],
                                        tr_suite,
                                        tr_bench))

        training_similarity.sort()

        results = {}

        if FLAGS.selection in ['elite', 'just']:
            # Get the most similar program
            training_similar_suite = training_similarity[0][1]
            training_similar_bench = training_similarity[0][2]

            # Load training sequences
            filename = '{}/{}/{}.yaml'.format(FLAGS.training_directory,
                                              training_similar_suite,
                                              training_similar_bench)
            sequences = IO.load_yaml_or_fail(filename)
            if FLAGS.selection == 'elite':
                sequences = filter_sequences(FLAGS.training_directory,
                                             training_similar_suite,
                                             training_similar_bench,
                                             FLAGS.levels_directory,
                                             FLAGS.compiler,
                                             FLAGS.level,
                                             sequences)

            # Rank the sequences
            rank = []
            for key, seq_data in sequences.items():
                rank.append((seq_data['goal'], seq_data['seq'], key))
            rank.sort()

            # Evaluate each sequence
            for _, sequence, key in rank:
                goal_value = Engine.evaluate(
                    Goals.prepare_goals(FLAGS.goals, FLAGS.weights),
                    Sequence.name_pass_to_string(sequence),
                    'opt',
                    bench_dir,
                    FLAGS.working_set,
                    FLAGS.times,
                    FLAGS.tool,
                    FLAGS.verify_output
                )
                if goal_value == float('inf'):
                    continue

                results[key] = {'seq': sequence, 'goal': goal_value}

                if len(results) == FLAGS.nof_sequences:
                    break

        elif FLAGS.selection == 'nearly':
            for _, training_suite, training_bench in training_similarity:
                # Load training sequences
                filename = '{}/{}/{}.yaml'.format(FLAGS.training_directory,
                                                  training_suite,
                                                  training_bench)
                sequences = IO.load_yaml_or_fail(filename)

                sequences = filter_sequences(FLAGS.training_directory,
                                             training_suite,
                                             training_bench,
                                             FLAGS.levels_directory,
                                             FLAGS.compiler,
                                             FLAGS.level,
                                             sequences)

                # Rank the sequences
                rank = []
                for key, seq_data in sequences.items():
                    rank.append((seq_data['goal'], seq_data['seq'], key))
                rank.sort()

                # Evaluate each sequence
                for _, sequence, key in rank:
                    goal_value = Engine.evaluate(
                        Goals.prepare_goals(FLAGS.goals, FLAGS.weights),
                        Sequence.name_pass_to_string(sequence),
                        'opt',
                        bench_dir,
                        FLAGS.working_set,
                        FLAGS.times,
                        FLAGS.tool,
                        FLAGS.verify_output
                    )
                    if goal_value == float('inf'):
                        continue

                    results[key] = {'seq': sequence, 'goal': goal_value}

                    if len(results) == FLAGS.nof_sequences:
                        break

                if len(results) == FLAGS.nof_sequences:
                    break

        if results:
            # Store the results
            IO.dump_yaml(results, results_filename)


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
                      'Goals')
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
    flags.DEFINE_enum('selection',
                      'elite',
                      ['elite', 'just', 'nearly'],
                      'Selection')
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
