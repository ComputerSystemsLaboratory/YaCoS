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
# Extract functions name from a program.
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.info import compy as R
from yacos.info.ncc import Inst2Vec

from yacos.essential import IO
from yacos.essential import Goal
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.essential import Similarity

from yacos.info.compy.extractors import LLVMDriver

from yacos.model import RepresentationExtractor
from yacos.model import GraphFromSequences


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # Verify the benchmark file
    if not os.path.isfile(FLAGS.benchmarks_filename):
        logging.error('File {} does not exist.'.format(
            FLAGS.benchmarks_filename)
        )
        sys.exit(1)

    # Verify benchmark directory
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)
    


    #check the representation type and set the function that will be extract it
    if FLAGS.representation_type == 'inst2vec':
        get_repr_function = RepresentationExtractor.get_inst2vec_features
    else:
        print('invalid representation')
        sys.exit(1)

    #create the valid data to graph input
    file_list = GraphFromSequences.create_file_list(FLAGS.benchmarks_directory,
                                             FLAGS.partial_compilation_directory,
                                             FLAGS.benchmarks_filename,
                                             get_repr_function)


    graph_fullpath = os.path.join(FLAGS.model_directory,
                                  FLAGS.graph_filename)

    representation_fullpath = os.path.join(FLAGS.model_directory,
                                           FLAGS.representation_filename)

    Model = GraphFromSequences(graph_fullpath ,
                               representation_fullpath)
    #Build model
    Model.build(file_list)

    # Create the output directory.
    os.makedirs(FLAGS.model_directory, exist_ok=True)
    
    #Save model
    Model.save()

# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'YAML Benchmarks Files')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('partial_compilation_directory',
                        None,
                        'Directory that contains the partial compilation files')
    flags.DEFINE_string('model_directory',
                        None,
                        'Directory that model will be stored')
    flags.DEFINE_string('graph_filename',
                        'model_graph.yaml',
                        'Filename that the graph of the model will be stored ')
    flags.DEFINE_string('representation_type',
                        'inst2vec',
                        'Representation used to benchmarks ')
    flags.DEFINE_string('representation_filename',
                        'model_representation.pck',
                        'Filename that the programs representation will be stored ')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('partial_compilation_directory')
    flags.mark_flag_as_required('model_directory')

    app.run(execute)
