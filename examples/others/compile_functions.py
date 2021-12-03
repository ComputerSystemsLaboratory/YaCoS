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
# Split a program into functions, compile each functions using
# diferent sequences, finally return the smallest code.
#

import os
import glob
import sys
import time
import traceback
import subprocess

from absl import app, flags, logging

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def run_command(cmdline, message=''):
    """Run a command."""
    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        sys.exit(1)


def optimize(work_dir, filename, sequence, out_filename=''):
    """Optimize a file."""
    sequence_str = Sequence.name_pass_to_string(sequence)
    cmdline = 'curr_dir=$PWD ; ' \
              + 'cd {0} ; ' \
              + 'opt {1} {2} -o {3} ; ' \
              + 'cd $curr_dir'
    cmdline = cmdline.format(work_dir,
                             sequence_str,
                             filename,
                             out_filename if out_filename else filename)

    run_command(cmdline, 'Compile: {}'.format(filename))


def execute(argv):
    """Generate inst2vec representation."""
    del argv

    # The sequences
    sequences = IO.load_yaml_or_fail(FLAGS.sequences_filename)

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)

    # Instantiate the LLVM driver.
    driver = LLVMDriver()
    # Instantiate the builder.
    builder = R.LLVMInstsBuilder(driver)

    for benchmark in benchmarks:
        try:
            compile_time = []

            # Verify if the benchmark was processed before
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

            data_filename = '{}/{}.yaml'.format(results_dir,
                                                bench_name)
            results_filename = '{}/{}_best.yaml'.format(results_dir,
                                                        bench_name)
            runtime_filename = '{}/{}_runtime.yaml'.format(results_dir,
                                                           bench_name)
            insts_filename = '{}/{}_instructions.yaml'.format(results_dir,
                                                              bench_name)
            ctime_filename = '{}/{}_compile_time.yaml'.format(results_dir,
                                                              bench_name)

            if FLAGS.verify_report and \
               os.path.isfile(data_filename) and \
               os.path.isfile(results_filename) and \
               os.path.isfile(insts_filename):
                continue

            # Split
            Engine.compile(benchmark_dir, 'opt', '-O0')

            data = Engine.extract_globals_and_functions('a.out_o.bc',
                                                        benchmark_dir,
                                                        True,
                                                        True)

            Engine.cleanup(benchmark_dir, 'opt')

            # Compile each function
            files = glob.glob('{}/*.ll'.format(benchmark_dir))

            results = {}
            for file_ in files:
                bitcode_name = file_.replace('{}/'.format(benchmark_dir), '')

                goal = []
                for _, seq_data in sequences.items():

                    # Output filename
                    out_filename = 'yacos.bc'

                    # Optimize
                    optimize(benchmark_dir,
                             bitcode_name,
                             seq_data['seq'],
                             out_filename)

                    # Extract the number of instructions
                    edata = builder.ir_to_info('{}/{}'.format(benchmark_dir,
                                                              out_filename))

                    if len(edata.functionInfos) > 1:
                        msg = 'Wrong number of funcs: {} - {} - {}.'.format(
                            benchmark,
                            file_,
                            edata.functionInfos[0].name
                        )
                        logging.error(msg)
                        sys.exit(1)

                    if not edata.functionInfos:
                        continue

                    goal.append((edata.functionInfos[0].instructions,
                                seq_data['seq']))

                if not goal:
                    continue

                goal.sort()

                results[bitcode_name.replace('.ll', '')] = {
                    'seq': goal[0][1],
                    'instructions': goal[0][0]
                }

            # Remove temporary bc files
            cmdline = 'rm -rf {}/yacos.bc'.format(benchmark_dir)
            run_command(cmdline, 'Remove temporary bc file')

            # Compile each function using the best sequence
            functions = list(results.keys())
            for file_ in files:
                bitcode_name = file_.replace('{}/'.format(benchmark_dir), '')
                bitcode_name = bitcode_name.replace('.ll', '')
                if bitcode_name in functions:
                    start_time = time.time()
                    optimize(benchmark_dir,
                             '{}.ll'.format(bitcode_name),
                             results[bitcode_name]['seq'])
                    end_time = time.time()
                    compile_time.append(end_time - start_time)
                else:
                    optimize(benchmark_dir,
                             '{}.ll'.format(bitcode_name),
                             '')

            # Generate target code
            start_time = time.time()
            Engine.compile(benchmark_dir, 'merge', '-O0')
            end_time = time.time()
            compile_time.append(end_time - start_time)

            # Extract the number of instructions
            edata = builder.ir_to_info('{}/{}'.format(benchmark_dir,
                                                      'a.out_o.bc'))
            insts = {}
            for item in edata.functionInfos:
                insts[item.name] = item.instructions

            if FLAGS.verify_integrity:
                # Execute the benchmark to validate the compilation
                goal_value = Engine.only_evaluate(
                        goals={'runtime': 1.0},
                        benchmark_directory=benchmark_dir,
                        times=3,
                        verify_output=True,
                        cleanup=False
                )

                # Store the runtime
                IO.dump_yaml([goal_value], runtime_filename)

            # Store the split data
            IO.dump_yaml(data, data_filename)

            # Store the best sequences
            IO.dump_yaml(results, results_filename)

            # Store the number of instructions
            IO.dump_yaml(insts, insts_filename)

            # Store the compile time
            IO.dump_yaml([sum(compile_time)], ctime_filename)

            # Cleanup benchmark directory
            Engine.cleanup(benchmark_dir, 'merge')

        except Exception:
            print('ERRO:', benchmark)
            traceback.print_exc(file=sys.stdout)
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
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences filename')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_list('sequence',
                      ['-O0'],
                      'Sequence')
    flags.DEFINE_boolean('verify_report',
                         False,
                         'Do not process the benchmark if a report exists')
    flags.DEFINE_boolean('verify_integrity',
                         False,
                         'Execute the program to validate the compilation')

    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('sequences_filename')
    flags.mark_flag_as_required('results_directory')

    FLAGS = flags.FLAGS

    app.run(execute)
