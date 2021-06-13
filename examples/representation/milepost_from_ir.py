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
# Extract Milepost static representation.
#

import os
import sys
import numpy as np

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Engine
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def numpy_data(data):
    """Dictionary to numpy."""
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
    functions = []
    for func, msf in data.items():
        values.append([msf[key] for key in keys])
        functions.append(func)

    return functions, values


def extract(benchmark_dir):
    """Extract the representation from the source code."""
    # Instantiate the LLVM driver.
    driver = LLVMDriver()
    # Instantiate the builder.
    builder = R.LLVMMSFBuilder(driver)

    # Compile de benchmark
    Engine.compile(benchmark_dir, 'opt', '-O0')

    # Read the bc file.
    source = '{}/a.out_o.bc'.format(benchmark_dir)

    if not os.path.isfile(source):
        return False

    # Extract the features.
    info = {}
    extractionInfo = builder.ir_to_info(source)
    for data in extractionInfo.functionInfos:
        info[data.name] = data.features

    # Cleanup de benchmark
    Engine.cleanup(benchmark_dir, 'opt')

    return info


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory.
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Process each benchmark.
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                     suite_name,
                                     bench_name)

        if not os.path.isdir(benchmark_dir):
            continue

        # Create the output directory.
        results_dir = os.path.join(FLAGS.results_directory, suite_name)
        os.makedirs(results_dir, exist_ok=True)

        filename = '{}/{}.npz'.format(results_dir, bench_name)
        if FLAGS.verify_report and os.path.isfile(filename):
            continue

        info = extract(benchmark_dir)
        if info:
            functions, values = numpy_data(info)
            np.savez_compressed(filename.replace('.npz', ''),
                                functions=functions,
                                values=values)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
