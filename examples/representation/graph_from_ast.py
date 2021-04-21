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
# Extract a graph representation from the whole module.
#
# The graphs are created from the AST.
#

import os
import sys
import glob

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.info import compy as R
from yacos.info.compy.extractors import ClangDriver


def extract(benchmark_dir,
            builder,
            graph_type):
    """Extract the representation from the source code."""
    # Define the visitor.
    visitors = {'ast': R.ASTVisitor,
                'astdata': R.ASTDataVisitor,
                'astdatacfg': R.ASTDataCFGVisitor}

    # Read the source codes.
    sources = glob.glob('{}/*.c'.format(benchmark_dir))
    representation = {}
    # Process each one.
    for source in sources:
        module = source.replace('{}/'.format(benchmark_dir), '')
        module = module.replace('.c', '')
        # Extract "information" from the source code (here data to construct
        # a graph).
        extractionInfo = builder.source_to_info(source)
        # Build the graph.
        representation[
                module
                      ] = builder.info_to_representation(extractionInfo,
                                                         visitors[graph_type])

    return representation


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks,
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory,
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Instantiate the Clang driver.
    driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O0,
        [],
        ["-xcl"]
    )

    # Define the builder
    builder = R.ASTGraphBuilder(driver)

    # Process each benchmark,
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                     suite_name,
                                     bench_name)

        if not os.path.isdir(benchmark_dir):
            continue

        # Extract the representation.
        representation = extract(benchmark_dir,
                                 builder,
                                 FLAGS.representation)

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite_name)

        # Create the output directory.
        os.makedirs(results_dir, exist_ok=True)

        # Store the representation.
        for module, graph in representation.items():
            filename = '{}/{}.pk'.format(results_dir, module)
            IO.dump_pickle(graph.networkx(), filename)

            if FLAGS.dot:
                # Store a dot.
                dot = graph.draw().decode()
                dotname = filename.replace('.pk', '.dot')
                fout = open(dotname, 'w')
                fout.write(dot)
                fout.close()


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_enum('representation',
                      'ast',
                      ['ast',  'astdata', 'astdatacfg'],
                      'Representation')
    flags.DEFINE_boolean('dot',
                         False,
                         'Save a dot file')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
