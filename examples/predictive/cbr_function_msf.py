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
# Compile each function using a specific sequence, taken from the most
# similar training program (function).
#
# This version extracts features.
#
# This version use:
#    - yacos.algorithm.CBR
#    - Milepost Static Features
#

import os
import sys
import glob

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.essential import Engine
from yacos.essential import Similarity
from yacos.algorithm import CBR_Function
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def sequences(training_directory):
    """Extract training sequences."""
    filenames = glob.glob('{}/**/*.yaml'.format(training_directory),
                          recursive=True)

    data = {}
    for filename in filenames:
        name = filename.replace('{}/'.format(training_directory), '')
        name = name.replace('.yaml', '')
        data[name] = IO.load_yaml(filename)

    return data


def extract_representation(benchmark_directory):
    """Extract Milepost Static Features"""
    # Instantiate the LLVM driver.
    driver = LLVMDriver()
    # Instantiate the builder.
    builder = R.LLVMMSFBuilder(driver)
    # The source files.
    source = '{}/a.out_o.bc'.format(benchmark_directory)

    # Compile the program
    Engine.compile(benchmark_directory, 'opt', '-O0')

    # Extract the features
    extractionInfo = builder.ir_to_info(source)

    # Cleanup the benchmark directory
    Engine.cleanup(benchmark_directory, 'opt')

    # Build a dictionary
    features = {}
    for data in extractionInfo.functionInfos:
        features[data.name] += data.features

    return features


def similarity(benchmark,
               benchmarks_directory,
               representation_directory):
    """Measure the similarity between two programs."""
    # Set the benchmark directory
    index = benchmark.find('.')
    bench_dir = os.path.join(benchmarks_directory,
                             benchmark[:index],
                             benchmark[index+1:])

    # Extract the representation for each function
    test_features = extract_representation(bench_dir)

    # Load the name of the functions
    filenames = glob.glob('{}/**/*_summary.yaml'.format(
                                                    representation_directory
                                                ),
                          recursive=True)

    # Load the training programs' features
    training_features = {}
    for filename in filenames:
        name = filename.replace('{}/'.format(representation_directory), '')
        name = name.replace('_summary.yaml', '')
        training_features[name] = IO.load_yaml(filename)

    # Measure the similarity
    training, test, similarity_ = Similarity.euclidean_distance_from_data(
                                    training_features,
                                    test_features
                                )

    # Return the similarity
    results = {}
    for i, function in enumerate(test):
        if function not in results:
            results[function] = {}
        for j, program in enumerate(training):
            results[function][program] = similarity_[i][j]

    return results


def execute(argv):
    """Predictive compilation."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Initialize a CBR object
    cbr = CBR_Function(Goal.prepare_goal(FLAGS.goals, FLAGS.weights),
                       'opt',
                       FLAGS.benchmarks_directory,
                       FLAGS.working_set,
                       FLAGS.times,
                       FLAGS.tool,
                       FLAGS.verify_output)

    # Process each benchmark
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

        # Create the results directory for the suite
        os.makedirs(results_dir, exist_ok=True)

        filename = '{}/{}.yaml'.format(results_dir, bench)

        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        training_sequences = sequences(FLAGS.training_directory)
        training_similarity = similarity(benchmark,
                                         FLAGS.benchmarks_directory,
                                         FLAGS.representation_directory)

        # Run CBR
        cbr.run(benchmark,
                training_sequences,
                training_similarity,
                FLAGS.nof_sequences)

        if cbr.results:
            # Store the results
            IO.dump_yaml(cbr.results, filename)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('levels_directory',
                        None,
                        'Levels directory')
    flags.DEFINE_string('representation_directory',
                        None,
                        'Representation directory')
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
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('levels_directory')
    flags.mark_flag_as_required('representation_directory')
    flags.mark_flag_as_required('training_directory')

    app.run(execute)
