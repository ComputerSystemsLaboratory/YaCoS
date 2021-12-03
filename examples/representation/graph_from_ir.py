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
# The graphs are created from IR.
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
    visitors = {
                # Clang
                'ast': R.ASTVisitor,
                'ast_data': R.ASTDataVisitor,
                'ast_data_cfg': R.ASTDataCFGVisitor,
                # LLVM
                # CFG
                'cfg': R.LLVMCFGVisitor,
                'cfg_compact': R.LLVMCFGCompactVisitor,
                'cfg_call': R.LLVMCFGCallVisitor,
                'cfg_call_nr': R.LLVMCFGCallNoRootVisitor,
                'cfg_call_compact_me': R.LLVMCFGCallCompactMultipleEdgesVisitor,
                'cfg_call_compact_se': R.LLVMCFGCallCompactSingleEdgeVisitor,
                'cfg_call_compact_me_nr': R.LLVMCFGCallCompactMultipleEdgesNoRootVisitor,
                'cfg_call_compact_se_nr': R.LLVMCFGCallCompactSingleEdgeNoRootVisitor,
                # CDFG
                'cdfg': R.LLVMCDFGVisitor,
                'cdfg_compact_me': R.LLVMCDFGCallCompactMultipleEdgesVisitor,
                'cdfg_compact_se': R.LLVMCDFGCallCompactSingleEdgeVisitor,
                'cdfg_call': R.LLVMCDFGCallVisitor,
                'cdfg_call_nr': R.LLVMCDFGCallNoRootVisitor,
                'cdfg_call_compact_me': R.LLVMCDFGCallCompactMultipleEdgesVisitor,
                'cdfg_call_compact_se': R.LLVMCDFGCallCompactSingleEdgeVisitor,
                'cdfg_call_compact_me_nr': R.LLVMCDFGCallCompactMultipleEdgesNoRootVisitor,
                'cdfg_call_compact_se_nr': R.LLVMCDFGCallCompactSingleEdgeNoRootVisitor,
                # CDFG PLUS
                'cdfg_plus': R.LLVMCDFGPlusVisitor,
                'cdfg_plus_nr': R.LLVMCDFGPlusNoRootVisitor,
                # PROGRAML
                'programl': R.LLVMProGraMLVisitor,
                'programl_nr': R.LLVMProGraMLNoRootVisitor
                }

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
    representation = builder.info_to_representation(extractionInfo,
                                                    visitors[graph_type])

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

    # Instantiate the LLVM driver.
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
                                 FLAGS.graph)

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite_name)

        # Create the output directory.
        os.makedirs(results_dir, exist_ok=True)

        # Store the representation.
        filename = '{}/{}.pk'.format(results_dir, bench_name)
        IO.dump_pickle(representation.networkx(), filename)

        if FLAGS.dot:
            # Store a dot.
            representation.draw(path=filename.replace('.pk', '.dot'))


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
    flags.DEFINE_enum('graph',
                      'programl_nr',
                      [
                      # Clang
                      'ast',
                      'ast_data',
                      'ast_data_cfg',
                      # LLVM
                      # CFG
                      'cfg',
                      'cfg_compact',
                      'cfg_call',
                      'cfg_call_nr',
                      'cfg_call_compact_me',
                      'cfg_call_compact_se',
                      'cfg_call_compact_me_nr',
                      'cfg_call_compact_se_nr',
                      # CDFG
                      'cdfg',
                      'cdfg_compact_me',
                      'cdfg_compact_se',
                      'cdfg_call',
                      'cdfg_call_nr',
                      'cdfg_call_compact_me',
                      'cdfg_call_compact_se',
                      'cdfg_call_compact_me_nr',
                      'cdfg_call_compact_se_nr',
                      # CDFG PLUS
                      'cdfg_plus',
                      'cdfg_plus_nr',
                      # PROGRAML
                      'programl',
                      'programl_nr'
                      ],
                      'The type of the graph')
    flags.DEFINE_boolean('dot',
                         False,
                         'Save a dot file')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
