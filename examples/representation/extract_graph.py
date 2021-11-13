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
# Classify applications into 104 classes given their raw code.
#
# The representation (graph) is created from IR.
#

import os
import sys
import glob
import numpy as np

from absl import app, flags, logging

from yacos.info import compy as R
from yacos.info.compy.extractors import ClangDriver, LLVMDriver


def extract_graph_data(graph, graph_type):
    """Convert the graph to StellarGraph representation."""

    nodes = {}

    if 'ast' in graph_type:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ast')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ast')
    else:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ir')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ir')
        nodes['i2v'] = graph.get_nodes_inst2vec_embeddings()
        nodes['ir2v'] = graph.get_nodes_ir2vec_embeddings()

    edges = graph.get_edges_str_dataFrame()

    return edges, nodes


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # Verify datset directory.
    if not os.path.isdir(FLAGS.dataset_directory):
        logging.error('Dataset directory {} does not exist.'.format(
            FLAGS.dataset_directory)
        )
        sys.exit(1)

    """Extract the representation from the source code."""

    if FLAGS.graph in ['ast', 'astdata', 'astdatacfg']:
        # Instantiate the Clang driver.
        driver = ClangDriver(
            ClangDriver.ProgrammingLanguage.C,
            ClangDriver.OptimizationLevel.O0,
            [],
            ["-xcl"]
        )
        # Define the builder
        builder = R.ASTGraphBuilder(driver)
    else:
        # Instantiate the LLVM driver.
        driver = LLVMDriver()
        # Define the builder
        builder = R.LLVMGraphBuilder(driver)

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

    folders = [
                os.path.join(FLAGS.dataset_directory, subdir)
                for subdir in os.listdir(FLAGS.dataset_directory)
                if os.path.isdir(os.path.join(FLAGS.dataset_directory, subdir))
              ]

    # Load data from all folders
    for folder in folders:
        sources = glob.glob('{}/*.ll'.format(folder))

        # Create the output directory.
        output_dir = '{}_{}'.format(folder, FLAGS.graph)
        os.makedirs(output_dir, exist_ok=True)

        for source in sources:
            # Extract "information" from the file
            # (data to construct the graph).
            if FLAGS.graph == 'asmcompact':
                print('not implemented yet')
                sys.exit(1)
            else:
                extractionInfo = builder.ir_to_info(source)
                # Build the graph.
                graph = builder.info_to_representation(extractionInfo,
                                                       visitors[FLAGS.graph])
            edges, nodes = extract_graph_data(graph, FLAGS.graph)

            filename = source.replace(folder, output_dir)
            filename = filename[:-3]
            np.savez_compressed(filename, edges=edges, nodes=nodes)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
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

    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
