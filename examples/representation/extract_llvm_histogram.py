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


def program_representation(functionInfos):
    """Find program representation."""
    keys = ['ret',
            'br',
            'switch',
            'indirectbr',
            'invoke',
            'callbr',
            'resume',
            'catchswitch',
            'catchret',
            'cleanupret',
            'unreachable',
            'fneg',
            'add',
            'fadd',
            'sub',
            'fsub',
            'mul',
            'fmul',
            'udiv',
            'sdiv',
            'fdiv',
            'urem',
            'srem',
            'frem',
            'shl',
            'lshr',
            'ashr',
            'and',
            'or',
            'xor',
            'extractelement',
            'insertelement',
            'sufflevector',
            'extractvalue',
            'insertvalue',
            'alloca',
            'load',
            'store',
            'fence',
            'cmpxchg',
            'atomicrmw',
            'getelementptr',
            'trunc',
            'zext',
            'sext',
            'fptrunc',
            'fpext',
            'fptoui',
            'fptosi',
            'uitofp',
            'sitofp',
            'ptrtoint',
            'inttoptr',
            'bitcast',
            'addrspacecast',
            'icmp',
            'fcmp',
            'phi',
            'select',
            'freeze',
            'call',
            'var_arg',
            'landingpad',
            'catchpad',
            'cleanuppad']

    values = []
    for data in functionInfos:
        values.append([data.instructions[key] for key in keys])

    return [sum(x) for x in zip(*values)]


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # Instantiate the LLVM driver.
    driver = LLVMDriver([])
    # Instantiate the builder.
    builder = R.LLVMHistogramBuilder(driver)

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

    # Load data from all folders
    for folder in folders:
        # Create the output directory.
        outdir = os.path.join(folder.replace(FLAGS.dataset_directory,
                              'llvm_histogram'))
        os.makedirs(outdir, exist_ok=True)

        # Extract "ir2vec" from the file
        sources = glob.glob('{}/*.ll'.format(folder))

        for source in sources:
            try:
                extractionInfo = builder.ir_to_info(source)
            except Exception:
                logging.error('Error {}.'.format(source))
                continue

            filename = source.replace(folder, outdir)
            filename = filename[:-3]
            np.savez_compressed(filename,
                                values=program_representation(
                                            extractionInfo.functionInfos
                                       ))


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
