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
# Extract names from a program (functions and global variables).
# Such task is used to extact and optimize a specific function.
#

import os
import sys
import glob

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.info import compy as R
from yacos.info.compy.extractors import ClangDriver


def process(benchmark_dir):
    """Transform Source code into representation."""
    # Instantiate the Clang Driver
    driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O3,
        [],
        ["-xcl"]
    )

    # Instantiate the builder
    builder = R.LLVMNamesBuilder(driver)

    # Read the source files
    sources = glob.glob('{}/*.c'.format(benchmark_dir))

    # Process each source file
    info = {}
    for source in sources:
        module = source.replace('{}/'.format(benchmark_dir), '')
        module = module.replace('.c', '')

        extractionInfo = builder.source_to_info(source)
        functions = [data.name for data in extractionInfo.functionInfos]
        globals_ = [data.name for data in extractionInfo.globalInfos]
        info[module] = (functions, globals_)

    return info


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify benchmark directory
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Process each benchmark
    info = {}
    for benchmark in tqdm(benchmarks, desc='Processing'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                     suite_name,
                                     bench_name)

        if not os.path.isdir(benchmark_dir):
            continue

        info[bench_name] = process(benchmark_dir)

    print(info)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')

    app.run(execute)
