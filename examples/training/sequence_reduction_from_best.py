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
# Create small sequences from the best one.
#

import os
import sys
import glob
import time

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.essential import Sequence
from yacos.algorithm import SequenceReduction


def execute(argv):
    """Create small sequence."""
    del argv

    FLAGS = flags.FLAGS

    # Verify directories.
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

    # The benchmarks.
    if FLAGS.benchmarks_filename:
        benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
        if not benchmarks:
            logging.exit('There are no benchmarks to process')
            sys.exit(1)
    else:
        benchmarks = glob.glob(
            '{}/*.yaml'.format(FLAGS.training_directory)
        )
        benchmarks = [
            b.replace(
                '{}/'.format(FLAGS.training_directory),
                '').replace(
                    '.yaml',
                    '')
            for b in benchmarks
        ]

    # Initialize a SequenceReduction object.
    seqred = SequenceReduction(Goal.prepare_goal(FLAGS.goals, FLAGS.weights),
                               'opt',
                               FLAGS.benchmarks_directory,
                               FLAGS.working_set,
                               FLAGS.times,
                               FLAGS.tool,
                               FLAGS.verify_output)

    # Process the benchmarks.
    runtime = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        results_dir = os.path.join(FLAGS.results_directory, suite_name)
        filename = '{}/{}.yaml'.format(results_dir, bench_name)

        # Create the output directory.
        os.makedirs(results_dir, exist_ok=True)

        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        bench_dir_training = os.path.join(
            FLAGS.training_directory,
            suite_name
        )
        filename_training = '{}/{}.yaml'.format(bench_dir_training, bench_name)

        # Load the sequences.
        sequences = IO.load_yaml(filename_training)
        if FLAGS.only_the_best:
            sequences = Sequence.get_the_best(sequences, FLAGS.nof_sequences)
        elif FLAGS.all_best:
            sequences = Sequence.get_all_best(sequences)

        # Reduce the sequences.
        results = {}
        counter = 0
        start_time = time.time()
        for _, data in sequences.items():
            seqred.run(data['seq'], benchmark)

            if not FLAGS.report_only_the_small:
                results[counter] = {'seq': seqred.results[0]['seq'],
                                    'goal': seqred.results[0]['goal']}
                counter += 1

            results[counter] = {'seq': seqred.results[1]['seq'],
                                'goal': seqred.results[1]['goal']}
            counter += 1

        end_time = time.time()

        # The runtime
        runtime[bench_name] = end_time - start_time

        # Store the results.
        IO.dump_yaml(results, filename)

    # Store the runtime
    filename = '{}/runtime.yaml'.format(results_dir)
    IO.dump_yaml(runtime, filename)


# Execute
if __name__ == '__main__':
    # APP
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
    flags.DEFINE_string('training_directory',
                        None,
                        'Training data directory')
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')
    flags.DEFINE_boolean('report_only_the_small',
                         False,
                         'Store only the small sequences')
    flags.DEFINE_boolean('all_best',
                         False,
                         'Evaluate all best sequences')
    flags.DEFINE_boolean('only_the_best',
                         True,
                         'Evaluate only the best sequence')
    flags.DEFINE_integer('nof_sequences',
                         1,
                         'Number of sequences to reduce (from each benchmark)')

    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('training_directory')
    flags.mark_bool_flags_as_mutual_exclusive(['all_best', 'only_the_best'])

    app.run(execute)
