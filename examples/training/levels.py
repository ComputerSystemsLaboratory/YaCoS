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
# Evaluate the compiler optimization levels.
#

import os
import sys
import time

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Goal
from yacos.essential import Engine


def execute(argv):
    """Evaluate the compiler optimization levels."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Process each benchmark.
    runtime = {}
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

        # Create the output directory.
        os.makedirs(results_dir, exist_ok=True)

        filename = '{}/{}.yaml'.format(results_dir, bench)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        results = {}
        start_time = time.time()

        # Process each compiler optimization level.
        for compiler in ['opt', 'llvm']:
            for level in FLAGS.levels:
                goal_value = Engine.evaluate(
                    Goal.prepare_goal(FLAGS.goals, FLAGS.weights),
                    '-{}'.format(level),
                    compiler,
                    bench_dir,
                    FLAGS.working_set,
                    FLAGS.times,
                    FLAGS.tool,
                    FLAGS.verify_output
                )
                compiler_name = 'clang' if compiler == 'llvm' else 'opt'
                if compiler_name not in results:
                    results[compiler_name] = {}
                results[compiler_name][level] = {'goal': goal_value,
                                                 'seq': ['-{}'.format(level)]}

        end_time = time.time()

        # The runtime
        runtime[bench] = end_time - start_time

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
    flags.DEFINE_list('levels',
                      ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz'],
                      'Compiler optimization levels')

    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
