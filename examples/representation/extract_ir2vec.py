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
from yacos.info.compy.extractors import LLVMDriver


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # Instantiate the LLVM driver.
    driver = LLVMDriver([])
    # Instantiate the builder.
    builder = R.LLVMIR2VecBuilder(driver)

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

    # Extract ir2vec
    ir2vec = {}
    max_length = []

    # Load data from all folders
    for folder in folders:
        # Create the output directory.
        output_dir = '{}_ir2vec'.format(folder)
        os.makedirs(output_dir, exist_ok=True)

        # Extract "ir2vec" from the file
        sources = glob.glob('{}/*.ll'.format(folder))

        for source in sources:
            extractionInfo = builder.ir_to_info(source)
            filename = source.replace(folder, output_dir)
            filename = filename[:-3]
            if FLAGS.embeddings == 'program':
                np.savez_compressed(filename,
                                    values=extractionInfo.moduleInfo.ir2vec)
            elif FLAGS.embeddings == 'instructions':
                ir2vec[filename] = extractionInfo.instructionInfos
                max_length.append(len(extractionInfo.instructionInfos))

    if FLAGS.embeddings == 'instructions':
        # Padding
        max_length = max(max_length)
        unknown = np.zeros(300)

        for filename, instructions in ir2vec.items():
            padding = []
            for instruction in instructions:
                padding.append(instruction.ir2vec)
            for i in range(len(instructions), max_length):
                padding.append(unknown)
            np.savez_compressed(filename, values=padding)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_enum('embeddings',
                      'program',
                      ['program', 'instructions'],
                      'Type of embeddings')
    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
