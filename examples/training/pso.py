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
# Apply Particle Swarm Optimization to generate sequences.
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Goals
from yacos.algorithms import PSO


def execute(argv):
    """Generate genetic sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks
    benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Initialize a SGA object
    pso = PSO(FLAGS.generations,
              FLAGS.population,
              FLAGS.omega,
              FLAGS.eta1,
              FLAGS.eta2,
              FLAGS.max_vel,
              FLAGS.variant,
              FLAGS.neighb_type,
              FLAGS.neighb_param,
              FLAGS.memory,
              FLAGS.seed,
              FLAGS.dimension,
              FLAGS.passes_filename,
              Goals.prepare_goals(FLAGS.goals, FLAGS.weights),
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

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite)

        # Create the results directory for the suite
        os.makedirs(results_dir, exist_ok=True)

        filename = '{}/{}.yaml'.format(results_dir, bench)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        pso.run(benchmark)

        if pso.results:
            IO.dump_yaml(pso.results,
                         filename,
                         FLAGS.report_only_the_best)


# Execute
if __name__ == '__main__':
    # APP
    flags.DEFINE_list('goals',
                      None,
                      'Goals')
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
    # PSO
    flags.DEFINE_integer('generations',
                         100,
                         'Number of generations')
    flags.DEFINE_integer('seed',
                         None,
                         'The seed')
    flags.DEFINE_integer('dimension',
                         100,
                         'Poblem dimension (individual length)')
    flags.DEFINE_integer('population',
                         100,
                         'Population size')
    flags.DEFINE_float('omega',
                       0.7298,
                       'Inertia weight (or constriction factor)')
    flags.DEFINE_float('eta1',
                       2.05,
                       'Social component')
    flags.DEFINE_float('eta2',
                       2.05,
                       'Cognitive component')
    flags.DEFINE_float('max_vel',
                       0.5,
                       'Maximum allowed particle velocities')
    flags.DEFINE_integer('variant',
                         5,
                         'Algorithmic variant')
    flags.DEFINE_integer('neighb_type',
                         2,
                         'Swarm topology')
    flags.DEFINE_integer('neighb_param',
                         4,
                         'Topology parameter')
    flags.DEFINE_bool('memory',
                      False,
                      'When true the velocities are not reset between calls')
    flags.DEFINE_string('passes_filename',
                        None,
                        'Filename (yaml) that describes the passes to use')

    flags.mark_flag_as_required('passes_filename')
    flags.mark_flag_as_required('goals')
    flags.mark_flag_as_required('weights')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
