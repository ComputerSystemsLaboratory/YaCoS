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
# Extract a graph representation from the whole module.
#
# The graphs are created from IR.
#

import os
import sys

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def process(benchmark_dir,
            sequences):
    """Transform Source code into representation."""
    # The builder and the driver
    builder = R.LLVMGraphBuilder(LLVMDriver())

    data = {}

    for key, seq_data in sequences.items():
        Engine.compile(benchmark_dir,
                        'opt',
                        Sequence.name_pass_to_string(seq_data['seq']))

        # Extract "information" from the IR
        filename = '{}/a.out_o.bc'.format(benchmark_dir)
        info = builder.ir_to_info(filename)

        # Process each function
        instructions = {}
        costs = {}
        for functionInfo in info.functionInfos:
            insts = [
                    len(bb.instructions)
                    for bb in functionInfo.basicBlocks
                    ]
            cost_size = [
                        inst.codeSize
                        for bb in functionInfo.basicBlocks
                        for inst in bb.instructions
                        ]
            instructions[functionInfo.name] = sum(insts)
            costs[functionInfo.name] = sum(cost_size)

        # Binary size
        filename = '{}/binary_size.yaml'.format(benchmark_dir)
        binary_size = IO.load_yaml(filename)

        # Code size
        filename = '{}/code_size.yaml'.format(benchmark_dir)
        code_size = IO.load_yaml(filename)

        # Results
        data[key] = {'instructions': instructions,
                     'cost_code_size': costs,
                     'binary_size': binary_size,
                     'code_size': code_size}

        # Clenaup
        Engine.cleanup(benchmark_dir, 'opt')

    return data


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks.
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # The sequences
    sequences = IO.load_yaml_or_fail(FLAGS.sequences_filename)
    if not sequences:
        logging.error('There are no sequences to process')
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

        # Extract the costs
        data = process(benchmark_dir, sequences)

        # Create the output directory.
        results_dir = os.path.join(FLAGS.results_directory,
                                   suite_name)
        os.makedirs(results_dir, exist_ok=True)

        # Store the results
        filename = '{}/{}.yaml'.format(results_dir, bench_name)
        IO.dump_yaml(data, filename)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('sequences_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
