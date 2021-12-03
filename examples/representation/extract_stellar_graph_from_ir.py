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
import pandas as pd
import pickle as pk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stellargraph import StellarDiGraph
from absl import app, flags, logging

from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def extract_graph_data(graph, graph_type):
    """Extract edges, nodes and embeddings."""
    nodes = {}
    #nodes['word2vec'] = graph.get_nodes_word2vec_embeddings('ir')
    nodes['histogram'] = graph.get_nodes_histogram_embeddings('ir')
    nodes['inst2vec'] = graph.get_nodes_inst2vec_embeddings()
    nodes['ir2vec'] = graph.get_nodes_ir2vec_embeddings()
    nodes['opcode'] = graph.get_nodes_opcode_embeddings()

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

    # Instantiate the LLVM driver.
    driver = LLVMDriver()
    # Define the builder
    builder = R.LLVMGraphBuilder(driver)

    # Define the visitor
    visitors = {
        # CFG
        'cfg_call': R.LLVMCFGCallVisitor,
        'cfg_call_nr': R.LLVMCFGCallNoRootVisitor,
        'cfg_call_compact_me': R.LLVMCFGCallCompactMultipleEdgesVisitor,
        'cfg_call_compact_se': R.LLVMCFGCallCompactSingleEdgeVisitor,
        'cfg_call_compact_me_nr': R.LLVMCFGCallCompactMultipleEdgesNoRootVisitor,
        'cfg_call_compact_se_nr': R.LLVMCFGCallCompactSingleEdgeNoRootVisitor,
        # CDFG
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

    idx = FLAGS.dataset_directory.rfind('/')
    last_folder = FLAGS.dataset_directory[idx+1:]

    # Load data from all folders
    for folder in folders:
        sources = glob.glob('{}/*.ll'.format(folder))

        for source in sources:
            try:
                extractionInfo = builder.ir_to_info(source)
                graph = builder.info_to_representation(extractionInfo,
                                                       visitors[FLAGS.graph])
                edges, nodes_data = extract_graph_data(graph, FLAGS.graph)
            except Exception:
                logging.error('Error {}.'.format(source))
                continue

            for feat, feat_data in nodes_data.items():
                indexes = []
                embeddings = []
                for idx, _, emb in feat_data:
                    indexes.append(idx)
                    embeddings.append(emb)
                nodes = pd.DataFrame(embeddings, index=indexes)

                graph = StellarDiGraph(nodes=nodes,
                                       edges=edges,
                                       edge_type_column="type")

                outdir = os.path.join(
                    folder.replace(
                        last_folder,
                        '{}_{}_{}'.format(last_folder, FLAGS.graph, feat)
                    )
                )

                os.makedirs(outdir, exist_ok=True)

                filename = source.replace('{}/'.format(folder), '')
                filename = filename.replace('.ll', '.pk')
                filename = '{}/{}'.format(outdir, filename)

                fout = open(filename, 'wb')
                pk.dump(graph, fout)
                fout.close()


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_enum('graph',
                      'cdfg_call',
                      [
                        # CFG
                        'cfg_call',
                        'cfg_call_nr',
                        'cfg_call_compact_me',
                        'cfg_call_compact_se',
                        'cfg_call_compact_me_nr',
                        'cfg_call_compact_se_nr',
                        # CDFG
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
