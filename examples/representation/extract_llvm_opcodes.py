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
import random as rn
import numpy as np

from absl import app, flags, logging

from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def program_representation(functionInfos):
    """Find program representation."""
    embeddings = []
    max_length = []
    for data in functionInfos:
        func_embeddings = []
        for op in data.opcodes:
            rn.seed(op)
            func_embeddings.append(rn.random())
        embeddings.append(func_embeddings)
        max_length.append(len(func_embeddings))

    # Padding
    rn.seed('unknown')
    unknown = rn.random()
    max_length = max(max_length)
    padding_embeddings = []
    for embedding in embeddings:
        padding_embedding = embedding
        for i in range(len(embedding), max_length):
            padding_embedding.append(unknown)
        padding_embeddings.append(padding_embedding)

    return [sum(x) for x in zip(*padding_embeddings)]


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # Instantiate the LLVM driver.
    driver = LLVMDriver([])
    # Instantiate the builder.
    builder = R.LLVMOpcodesBuilder(driver)

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

    # Extract LLVM opcodes
    llvm_opcodes = {}
    max_length = []
    for folder in folders:
        llvm_opcodes[folder] = {}
        sources = glob.glob('{}/*.ll'.format(folder))
        for source in sources:
            extractionInfo = builder.ir_to_info(source)
            embeddings = program_representation(extractionInfo.functionInfos)
            bench_name = source.replace('{}/'.format(folder), '')
            bench_name = bench_name.replace('.ll', '')
            llvm_opcodes[folder][bench_name] = embeddings
            max_length.append(len(embeddings))

    # Padding
    rn.seed('unknown')
    unknown = rn.random()
    max_length = max(max_length)

    for folder, data in llvm_opcodes.items():
        # Create the output directory.
        output_dir = '{}_llvm_opcodes'.format(folder)
        os.makedirs(output_dir, exist_ok=True)

        for bench, embeddings in data.items():
            padding_embeddings = embeddings
            for i in range(len(embeddings), max_length):
                padding_embeddings.append(unknown)

            filename = os.path.join(output_dir, bench)
            np.savez_compressed(filename, values=padding_embeddings)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
