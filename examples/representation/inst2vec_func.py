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
# Extract inst2vec representation, for each function.
#

import os
import numpy as np

from absl import app, flags
from yacos.info.ncc import Inst2Vec
from yacos.essential import IO
from yacos.essential import Engine


def execute(argv):
    """Generate inst2vec representation."""
    del argv

    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)

    for benchmark in benchmarks:
        try:
            index = benchmark.find('.')
            suite_name = benchmark[:index]
            bench_name = benchmark[index+1:]

            benchmark_dir = os.path.join(FLAGS.benchmarks_directory,
                                         suite_name,
                                         bench_name)
            if not os.path.isdir(benchmark_dir):
                continue

            results_dir = os.path.join(FLAGS.results_directory, suite_name)
            os.makedirs(results_dir, exist_ok=True)

            filename = '{}/{}.npz'.format(results_dir, bench_name)
            if FLAGS.verify_report and os.path.isfile(filename):
                continue

            Engine.compile(benchmark_dir, 'opt', '-O0')

            data = Engine.extract_globals_and_functions('a.out_o.bc',
                                                        benchmark_dir,
                                                        False,
                                                        True)

            Inst2Vec.prepare_benchmark(FLAGS.benchmarks_directory, benchmark)

            rep = Inst2Vec.extract()

            functions = []
            inst2vec = []
            for func, i2v in rep.items():
                functions.append(func)
                inst2vec.append(np.array(i2v))

            np.savez_compressed(filename.replace('.npz', ''),
                                functions=functions,
                                values=inst2vec)

            IO.dump_yaml(data, filename.replace('.npz', '.yaml'))

            Engine.cleanup(benchmark_dir, 'opt')

            Inst2Vec.remove_data_directory()
        except Exception:
            print('ERRO:', benchmark)
            continue


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks filename')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_list('sequence',
                      ['-O0'],
                      'Sequence')
    flags.DEFINE_boolean('verify_report',
                         True,
                         'Do not process the benchmark if a report exists')

    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('results_directory')

    FLAGS = flags.FLAGS

    app.run(execute)
