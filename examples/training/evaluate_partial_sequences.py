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
# Evaluate partial sequences
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

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # The sequence
    sequences = IO.load_yaml_or_fail(FLAGS.sequences_filename)
    if not sequences:
        logging.error('There are no sequences to process')
        sys.exit(1)

    if FLAGS.all_best:
        sequences = Sequence.get_all_best(sequences)

    # Process each benchmark.
    runtime = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        # The benchmark.
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        bench_dir = os.path.join(FLAGS.benchmarks_directory,
                                 suite_name,
                                 bench_name)

        if not os.path.isdir(bench_dir):
            logging.error('Benchmark {} does not exist.'.format(benchmark))
            sys.exit(1)

        # Create the output directory.
        results_dir = os.path.join(FLAGS.results_directory,
                                   suite_name)
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
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences')

    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('sequences_filename')

    FLAGS = flags.FLAGS

    app.run(execute)
