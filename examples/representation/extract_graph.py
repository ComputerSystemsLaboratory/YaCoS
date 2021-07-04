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

    nodes = []

    if 'ast' in graph_type:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ast')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ast')
    elif 'asm' in graph_type:
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ir')
    else:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ir')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ir')
        nodes['i2v'] = graph.get_nodes_inst2vec_embeddings()
        nodes['ir2v'] = graph.get_nodes_ir2vec_embeddings()

    edges = graph.get_edges_dataFrame()
    adj = graph.get_adjacency_matrix()

    return adj, edges, nodes


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
                'astdata': R.ASTDataVisitor,
                'astdatacfg': R.ASTDataCFGVisitor,
                # LLVM
                'programl': R.LLVMProGraMLVisitor,
                'programlnoroot': R.LLVMProGraMLNoRootVisitor,
                'cfg': R.LLVMCFGVisitor,
                'cfgcompact': R.LLVMCFGCompactVisitor,
                'cfgcall': R.LLVMCFGCallVisitor,
                'cfgcallnoroot': R.LLVMCFGCallNoRootVisitor,
                'cfgcallcompact': R.LLVMCFGCallCompactVisitor,
                'cfgcallcompact1e': R.LLVMCFGCallCompactOneEdgeVisitor,
                'cfgcallcompactnoroot': R.LLVMCFGCallCompactNoRootVisitor,
                'cfgcallcompact1enoroot': R.LLVMCFGCallCompactOneEdgeNoRootVisitor,
                'cdfg': R.LLVMCDFGVisitor,
                'cdfgcompact': R.LLVMCDFGCompactVisitor,
                'cdfgcompact1e': R.LLVMCDFGCompactOneEdgeVisitor,
                'cdfgcall': R.LLVMCDFGCallVisitor,
                'cdfgcallnoroot': R.LLVMCDFGCallNoRootVisitor,
                'cdfgcallcompact': R.LLVMCDFGCallCompactVisitor,
                'cdfgcallcompact1e': R.LLVMCDFGCallCompactOneEdgeVisitor,
                'cdfgcallcompactnoroot': R.LLVMCDFGCallCompactNoRootVisitor,
                'cdfgcallcompact1enoroot': R.LLVMCDFGCallCompactOneEdgeNoRootVisitor,
                'cdfgplus': R.LLVMCDFGPlusVisitor,
                'cdfgplusnoroot': R.LLVMCDFGPlusNoRootVisitor
                }

    folders = [
                os.path.join(FLAGS.dataset_directory, subdir)
                for subdir in os.listdir(FLAGS.dataset_directory)
                if os.path.isdir(os.path.join(FLAGS.dataset_directory, subdir))
              ]
    print(folders)
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
            adj, edges, nodes = extract_graph_data(graph, FLAGS.graph)

            filename = source.replace(folder, output_dir)
            filename = filename[:-3]
            np.savez_compressed(filename, adj=adj, edges=edges, nodes=nodes)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_enum('graph',
                      'cdfgcallcompactnoroot',
                      [
                        'ast',
                        'astdata',
                        'astdatacfg',
                        'programl',
                        'programlnoroot',
                        'cfg',
                        'cfgcompact',
                        'cfgcall',
                        'cfgcallnoroot',
                        'cfgcallcompact',
                        'cfgcallcompact1e'
                        'cfgcallcompactnoroot',
                        'cfgcallcompact1enoroot',
                        'cdfg',
                        'cdfgcompact',
                        'cdfgcompact1e',
                        'cdfgcall',
                        'cdfgcallnoroot',
                        'cdfgcallcompact',
                        'cdfgcallcompact1e',
                        'cdfgcallcompactnoroot',
                        'cdfgcallcompact1enoroot',
                        'cdfgplus',
                        'cdfgplusnoroot',
                        'asmcompact'
                      ],
                      'The type of the graph')

    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
