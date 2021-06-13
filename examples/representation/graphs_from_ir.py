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
# Extract a graph representation from each function.
#
# IR graphs
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Engine
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def extract(benchmark_dir,
            builder,
            graph_type):
    """Extract the representation from the source code."""
    # Define the visitor
    visitors = {'programl': R.LLVMProGraMLVisitor,
                'programlnoroot': R.LLVMProGraMLNoRootVisitor,
                'cfg': R.LLVMCFGVisitor,
                'cfgcompact': R.LLVMCFGCompactVisitor,
                'cfgcall': R.LLVMCFGCallVisitor,
                'cfgcallnoroot': R.LLVMCFGCallNoRootVisitor,
                'cfgcallcompact': R.LLVMCFGCallCompactVisitor,
                'cfgcallcompactnoroot': R.LLVMCFGCallCompactNoRootVisitor,
                'cdfg': R.LLVMCDFGVisitor,
                'cdfgcompact': R.LLVMCDFGCompactVisitor,
                'cdfgcall': R.LLVMCDFGCallVisitor,
                'cdfgcallnoroot': R.LLVMCDFGCallNoRootVisitor,
                'cdfgcallcompact': R.LLVMCDFGCallCompactVisitor,
                'cdfgcallcompactnoroot': R.LLVMCDFGCallCompactNoRootVisitor,
                'cdfgplus': R.LLVMCDFGPlusVisitor,
                'cdfgplusnoroot': R.LLVMCDFGPlusNoRootVisitor}

    # Compile de benchmark
    Engine.compile(benchmark_dir, 'opt', '-O0')

    # Read the bc file.
    source = '{}/a.out_o.bc'.format(benchmark_dir)

    if not os.path.isfile(source):
        return False

    # Extract "information" from the file (here data to construct
    # a graph).
    extractionInfo = builder.ir_to_info(source)
    # Build the graph.
    representation = {}
    for functionInfo in extractionInfo.functionInfos:
        representation[
            functionInfo.name
        ] = builder.info_to_representation(functionInfo, visitors[graph_type])

    # Cleanup de benchmark
    Engine.cleanup(benchmark_dir, 'opt')

    return representation


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory.
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Instantiate the clang driver.
    driver = LLVMDriver()

    # Define the builder
    builder = R.LLVMGraphBuilder(driver)

    # Process each benchmark.
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
        for function, graph in representation.items():
            filename = '{}/{}_{}.pk'.format(results_dir, bench_name, function)
            IO.dump_pickle(graph.networkx(), filename)

            if FLAGS.dot:
                graph.draw(path=filename.replace('.pk', '.dot'))


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
                      'programl',
                      [
                        'programl',
                        'programlnoroot',
                        'cfg',
                        'cfgcompact',
                        'cfgcall',
                        'cfgcallnoroot',
                        'cfgcallcompact',
                        'cfgcallcompactnoroot',
                        'cdfg',
                        'cdfgcompact',
                        'cdfgcall',
                        'cdfgcallnoroot',
                        'cdfgcallcompact',
                        'cdfgcallcompactnoroot',
                        'cdfgplus',
                        'cdfgplusnoroot'
                      ],
                      'Representation')
    flags.DEFINE_boolean('dot',
                         False,
                         'Save a dot file')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
