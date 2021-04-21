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
# Extract IR2Vec representation.
#

import os
import sys
import numpy as np

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essentials import IO
from yacos.essentials import Engine
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def numpy_data(data):
    """Dictionary to numpy."""
    values = []
    functions = []
    for func, ir2vec in data.items():
        if func == 'program':
            continue
        functions.append(func)
        values.append(ir2vec)

    functions.append('program')
    values.append(data['program'])

    return functions, values


def extract(benchmark_dir):
    """Extract the representation from the source code."""
    # Instantiate the LLVM driver.
    driver = LLVMDriver([])
    # Instantiate the builder.
    builder = R.LLVMIR2VecBuilder(driver)

    # Compile de benchmark
    Engine.compile_(benchmark_dir, 'opt', '-O0')

    # Read the bc file.
    source = '{}/a.out_o.bc'.format(benchmark_dir)

    if not os.path.isfile(source):
        return False

    # Extract the features.
    info = {}
    try:
        extractionInfo = builder.ir_to_info(source)
        for data in extractionInfo.functionInfos:
            info[data.name] = data.ir2vec

        info['program'] = extractionInfo.moduleInfo.ir2vec
    except Exception:
        pass

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

        print(bench_name, flush=True)
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
