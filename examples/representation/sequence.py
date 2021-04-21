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
# Extract sequence representation.
#

import os
import sys
import glob

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.info import compy as R
from yacos.info.compy.extractors import ClangDriver


def extract(benchmark_dir,
            builder,
            visitor):
    """Extract the representation from the source code."""
    # Load source codes.
    sources = glob.glob('{}/*.c'.format(benchmark_dir))
    # Process each one.
    representation = {}
    for source in sources:
        module = source.replace('{}/'.format(benchmark_dir), '')
        module = module.replace('.c', '')
        representation[module] = {}
        # Extract "information" from the source code (here tokens to construct
        # a sequence).
        extractionInfo = builder.source_to_info(source)
        # Build the sequence.
        for functionInfo in extractionInfo.functionInfos:
            representation[module][
                functionInfo.name
            ] = builder.info_to_representation(functionInfo, visitor)

    return representation


def execute(argv):
    """Extract a sequence representation."""
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

    # Define the builder and the visitor.
    if FLAGS.representation == 'seq':
        builder = R.SyntaxSeqBuilder
        visitor = R.SyntaxSeqVisitor
    elif FLAGS.representation == 'token':
        builder = R.SyntaxSeqBuilder
        visitor = R.SyntaxTokenkindVisitor
    elif FLAGS.representation == 'tokenvar':
        builder = R.SyntaxSeqBuilder
        visitor = R.SyntaxTokenkindVariableVisitor
    elif FLAGS.representation == 'llvm':
        builder = R.LLVMSeqBuilder
        visitor = R.LLVMSeqVisitor

    # Instantiate the clang driver.
    driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O3,
        [],
        ["-xcl"],
    )

    # Process each benchmark
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                     suite_name,
                                     bench_name)

        if not os.path.isdir(benchmark_dir):
            continue

        # Extract the representation.
        representation = extract(benchmark_dir,
                                 builder(driver),
                                 visitor)

        results_dir = os.path.join(FLAGS.results_directory,
                                   suite_name)

        # Create the results directory for the suite
        os.makedirs(results_dir, exist_ok=True)

        for module, functions in representation.items():
            for function, rep in functions.items():
                filename = '{}/{}_{}.dot'.format(results_dir, module, function)
                dot = rep.draw().decode()
                fout = open(filename, 'w')
                fout.write(dot)
                fout.close()


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_enum('representation',
                      'seq',
                      ['seq',  'token', 'tokenvar', 'llvm'],
                      'Representation')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
