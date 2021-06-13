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
# Evaluate partial sequences from the good one.
#
# Example:
# -sroa -gvn -inline -mem2reg
#
# The algorithm will evaluate:
# -sroa
# -sroa -gvn
# -sroa -gvn -inline
# -sroa -gvn -inline -mem2reg
#

import os
import sys
import time

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.essential import Engine
from yacos.essential import Sequence


def execute(argv):
    """Execute the algorithm."""
    del argv

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Process each benchmark.
    runtime = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        # The benchmark.
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench_name = benchmark[index+1:]

        bench_dir = os.path.join(FLAGS.benchmarks_directory,
                                 suite,
                                 bench_name)

        if not os.path.isdir(bench_dir):
            logging.error('Benchmark {} does not exist.'.format(benchmark))
            sys.exit(1)

        # The training data.
        training_dir = os.path.join(FLAGS.training_directory,
                                    suite)
        filename = '{}/{}.yaml'.format(training_dir, bench_name)

        sequences = IO.load_yaml_or_fail(filename)
        if not sequences:
            logging.error('There are no sequences to process')
            sys.exit(1)

        best_sequence = Sequence.get_the_best(sequences)

        if FLAGS.baseline_directory:
            # Verify if the best sequence is better than the baseline.
            baseline_dir = os.path.join(FLAGS.baseline_directory,
                                        suite)
            filename = '{}/{}.yaml'.format(baseline_dir, bench_name)
            baseline_data = IO.load_yaml_or_fail(filename)
            if not baseline_data:
                logging.error('There are no baseline data')
                sys.exit(1)

            baseline_goal = baseline_data[
                                            FLAGS.compiler
                                         ][FLAGS.baseline]['goal']
            for _, data in best_sequence.items():
                best_sequence_goal = data['goal']

            if not (best_sequence_goal < baseline_goal):
                continue

        # Get the best sequences
        if FLAGS.all_best:
            sequences = Sequence.get_all_best(sequences)
        else:
            sequences = best_sequence

        # Create the output directory.
        results_dir = os.path.join(FLAGS.results_directory,
                                   suite)

        os.makedirs(results_dir, exist_ok=True)

        # Verify report.
        if FLAGS.suffix:
            filename = '{}/{}_{}.yaml'.format(
                results_dir,
                bench_name,
                FLAGS.suffix
            )
        else:
            filename = '{}/{}.yaml'.format(results_dir, bench_name)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        # Process the sequences
        results = {}
        start_time = time.time()
        for seq_key, seq_data in sequences.items():
            # Generate small sequences
            partial_sequences = Sequence.split(seq_data['seq'], FLAGS.expand)
            # Process the partial sequences.
            results[seq_key] = {}
            for partial_key, partial_seq_data in partial_sequences.items():
                goal_value = Engine.evaluate(
                    Goal.prepare_goal(FLAGS.goals, FLAGS.weights),
                    Sequence.name_pass_to_string(partial_seq_data['seq']),
                    'opt',
                    bench_dir,
                    FLAGS.working_set,
                    FLAGS.times,
                    FLAGS.tool,
                    FLAGS.verify_output
                )

                results[seq_key][partial_key] = {
                                'seq': partial_seq_data['seq'],
                                'goal': goal_value
                }

        end_time = time.time()

        # The runtime
        runtime[bench_name] = end_time - start_time

        # Store the results
        IO.dump_yaml(results, filename)

    # Store the runtime
    filename = '{}/runtime.yaml'.format(results_dir)
    IO.dump_yaml(runtime, filename)


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
    flags.DEFINE_boolean('all_best',
                         False,
                         'Evaluate all best sequences')
    flags.DEFINE_boolean('expand',
                         False,
                         'Expand the original sequence')
    flags.DEFINE_string('suffix',
                        '',
                        'Filename suffix')
    flags.DEFINE_string('baseline_directory',
                        None,
                        'Compiler optimization levels directory')
    flags.DEFINE_string('training_directory',
                        None,
                        'Training directory')
    flags.DEFINE_enum('baseline',
                      'Oz',
                      ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz'],
                      'Baseline')
    flags.DEFINE_enum('compiler',
                      'clang',
                      ['clang', 'opt'],
                      'Compiler')

    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('training_directory')

    FLAGS = flags.FLAGS

    app.run(execute)
