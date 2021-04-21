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
# Generate random sequences.
#

import os
import sys
import time

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Goals
from yacos.algorithms import Random


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory.
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Initialize a Random object.
    rnd = Random(FLAGS.nof_sequences,
                 FLAGS.minimum,
                 FLAGS.maximum,
                 FLAGS.factor,
                 FLAGS.ssa,
                 FLAGS.shuffle,
                 FLAGS.update,
                 FLAGS.repetition,
                 FLAGS.original,
                 FLAGS.passes_filename,
                 Goals.prepare_goals(FLAGS.goals, FLAGS.weights),
                 'opt',
                 FLAGS.benchmarks_directory,
                 FLAGS.working_set,
                 FLAGS.times,
                 FLAGS.tool,
                 FLAGS.verify_output)

    # Process each benchmark.
    runtime = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite = benchmark[:index]
        bench = benchmark[index+1:]

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite)

        # Create the results directory.
        os.makedirs(results_dir, exist_ok=True)

        filename = '{}/{}.yaml'.format(results_dir, bench)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        # Run the algorithm.
        start_time = time.time()
        rnd.run(benchmark)
        end_time = time.time()

        # The runtime
        runtime[bench] = end_time - start_time

        # Store the results.
        if rnd.results:
            IO.dump_yaml(rnd.results,
                         filename,
                         FLAGS.report_only_the_best)

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
    flags.DEFINE_boolean('report_only_the_best',
                         False,
                         'Store only the best result')
    flags.DEFINE_list('goals',
                      None,
                      'Goals')
    flags.DEFINE_list('weights',
                      None,
                      'Weights')
    flags.DEFINE_string('passes_filename',
                        None,
                        'Filename (yaml) that describes the passes to use')
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
    # RANDOM
    flags.DEFINE_integer('nof_sequences',
                         64,
                         'Number of sequences',
                         lower_bound=1)
    flags.DEFINE_integer('minimum',
                         10,
                         'Minimum sequence\'s length',
                         lower_bound=1)
    flags.DEFINE_integer('maximum',
                         100,
                         'Maximum sequence\'s length')
    flags.DEFINE_integer('factor',
                         1,
                         'Factor',
                         lower_bound=1)
    flags.DEFINE_boolean('ssa',
                         False,
                         'Use (add) mem2reg as the first pass')
    flags.DEFINE_boolean('shuffle',
                         False,
                         'Shuffle the sequence to generate a new one')
    flags.DEFINE_boolean('update',
                         True,
                         'Update the new sequence')
    flags.DEFINE_boolean('repetition',
                         True,
                         'Allow a pass to be used multiple times')
    flags.DEFINE_boolean('original',
                         False,
                         'Store the original sequence')

    flags.mark_flag_as_required('passes_filename')
    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
