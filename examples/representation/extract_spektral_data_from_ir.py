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
# Extract Spektral data.
#

import os
import sys
import glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import app, flags, logging

from spektral.utils.sparse import reorder

from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def prepare_node_features(nodes_embeddings):
    """Extract the node features."""
    features = {idx:feat for (idx, _, feat) in nodes_embeddings}

    return np.array([features[key] for key in sorted(features.keys())])
    
def extract_graph_data(graph, graph_type):
    """Extract edges, nodes and embeddings."""

    a = graph.get_adjacency_matrix()

    x = {}
    x['histogram'] = prepare_node_features(graph.get_nodes_histogram_embeddings('ir'))
    x['inst2vec'] = prepare_node_features(graph.get_nodes_inst2vec_embeddings())
    x['ir2vec'] = prepare_node_features(graph.get_nodes_ir2vec_embeddings())
    x['opcode'] = prepare_node_features(graph.get_nodes_opcode_embeddings())

    edge_index, edge_features = graph.get_edges_histogram_embeddings()

    e = reorder(edge_index, edge_features=edge_features)

    return a, x, e[1]

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

    outdir = '{}_spektral'.format(FLAGS.dataset_directory)
    os.makedirs(outdir, exist_ok=True)

    # Load data from all folders
    for folder in folders:
        sources = glob.glob('{}/*.ll'.format(folder))

        for source in sources:
            try:
                extractionInfo = builder.ir_to_info(source)
                graph = builder.info_to_representation(extractionInfo,
                                                       visitors[FLAGS.graph])
                a, x, e = extract_graph_data(graph, FLAGS.graph)
            except Exception:
                logging.error('Error {}.'.format(source))
                continue

            idx = folder.rfind('/')
            label = folder[idx+1:]
    
            y = [0 for _ in range(len(folders))]
            y[int(label)] = 1

            filename = source.replace('{}/'.format(folder), '')
            filename = filename.replace('.ll', '')
            filename = '{}/{}_{}'.format(outdir, label, filename)

            np.savez_compressed(filename,
                                a=a,
                                x_histogram=x['histogram'],
                                x_inst2vec=x['inst2vec'],
                                x_ir2vec=x['ir2vec'],
                                x_opcode=x['opcode'],
                                e=e,
                                y=y)
            

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
