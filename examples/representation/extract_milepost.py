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
    keys = ['ft01_BBInMethod',
            'ft02_BBWithOneSuccessor',
            'ft03_BBWithTwoSuccessors',
            'ft04_BBWithMoreThanTwoSuccessors',
            'ft05_BBWithOnePredecessor',
            'ft06_BBWithTwoPredecessors',
            'ft07_BBWithMoreThanTwoPredecessors',
            'ft08_BBWithOnePredOneSuc',
            'ft09_BBWithOnePredTwoSuc',
            'ft10_BBWithTwoPredOneSuc',
            'ft11_BBWithTwoPredTwoSuc',
            'ft12_BBWithMoreTwoPredMoreTwoSuc',
            'ft13_BBWithInstructionsLessThan15',
            'ft14_BBWithInstructionsIn[15-500]',
            'ft15_BBWithInstructionsGreaterThan500',
            'ft16_EdgesInCFG',
            'ft17_CriticalEdgesInCFG',
            'ft18_AbnormalEdgesInCFG',
            'ft19_DirectCalls',
            'ft20_ConditionalBranch',
            'ft21_AssignmentInstructions',
            'ft22_ConditionalBranch',
            'ft23_BinaryIntOperations',
            'ft24_BinaryFloatPTROperations',
            'ft25_Instructions',
            'ft26_AverageInstruction',
            'ft27_AveragePhiNodes',
            'ft28_AverageArgsPhiNodes',
            'ft29_BBWithoutPhiNodes',
            'ft30_BBWithPHINodesIn[0-3]',
            'ft31_BBWithMoreThan3PHINodes',
            'ft32_BBWithArgsPHINodesGreaterThan5',
            'ft33_BBWithArgsPHINodesGreaterIn[1-5]',
            'ft34_SwitchInstructions',
            'ft35_UnaryOperations',
            'ft36_InstructionThatDoPTRArithmetic',
            'ft37_IndirectRefs',
            'ft38_AdressVarIsTaken',
            'ft39_AddressFunctionIsTaken',
            'ft40_IndirectCalls',
            'ft41_AssignmentInstructionsWithLeftOperandIntegerConstant',
            'ft42_BinaryOperationsWithOneOperandIntegerConstant',
            'ft43_CallsWithPointersArgument',
            'ft44_CallsWithArgsGreaterThan4',
            'ft45_CallsThatReturnPTR',
            'ft46_CallsThatReturnInt',
            'ft47_ConstantZero',
            'ft48_32-bitIntegerConstants',
            'ft49_ConstantOne',
            'ft50_64-bitIntegerConstants',
            'ft51_ReferencesLocalVariables',
            'ft52_DefUseVariables',
            'ft53_LocalVariablesReferred',
            'ft54_ExternVariablesReferred',
            'ft55_LocalVariablesPointers',
            'ft56_VariablesPointers']

    values = []
    for data in functionInfos:
        values.append([data.features[key] for key in keys])

    return [sum(x) for x in zip(*values)]


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # Instantiate the LLVM driver.
    driver = LLVMDriver([])
    # Instantiate the builder.
    builder = R.LLVMMSFBuilder(driver)

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
                              'milepost'))
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
