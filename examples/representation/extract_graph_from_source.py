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
from yacos.info.compy.extractors import ClangDriver


def extract_graph_data(graph, graph_type):
    """Convert the graph to StellarGraph representation."""

    nodes = {}

    nodes['word2vec'] = graph.get_nodes_word2vec_embeddings('ast')
    nodes['histogram'] = graph.get_nodes_histogram_embeddings('ast')

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

    folders = [
                os.path.join(FLAGS.dataset_directory, subdir)
                for subdir in os.listdir(FLAGS.dataset_directory)
                if os.path.isdir(os.path.join(FLAGS.dataset_directory, subdir))
              ]

    idx = FLAGS.dataset_directory.rfind('/')
    last_folder = FLAGS.dataset_directory[idx+1:]

    # Define the visitor
    visitors = {
                'ast': R.ASTVisitor,
                'ast_data': R.ASTDataVisitor,
                'ast_data_cfg': R.ASTDataCFGVisitor
                }

    # Load data from all folders
    for folder in folders:
        # Create the output directory.
        output_dir = os.path.join(folder.replace(last_folder,
                                                 '{}_{}'.format(last_folder,
                                                                FLAGS.graph)))

        os.makedirs(output_dir, exist_ok=True)

        sources = []
        types = ['{}/*.c'.format(folder), '{}/*.cpp'.format(folder)]
        for files in types:
            sources.extend(glob.glob(files))

        for source in sources:
            # Instantiate the Clang driver.
            if source.endswith('.c'):
                driver = ClangDriver(
                    ClangDriver.ProgrammingLanguage.C,
                    ClangDriver.OptimizationLevel.O0,
                    [],
                    ["-xcl"]
                )
            elif source.endswith('.cpp'):
                driver = ClangDriver(
                    ClangDriver.ProgrammingLanguage.CPlusPlus,
                    ClangDriver.OptimizationLevel.O0,
                    [],
                    ["-xcl"]
                )
            else:
                logging.error('File type error: {}.'.format(source))
                sys.exit(1)

            # Define the builder
            builder = R.ASTGraphBuilder(driver)

            extractionInfo = builder.source_to_info(source)
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
                      'ast',
                      [
                        'ast',
                        'ast_data',
                        'ast_data_cfg',
                      ],
                      'The type of the graph')

    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
