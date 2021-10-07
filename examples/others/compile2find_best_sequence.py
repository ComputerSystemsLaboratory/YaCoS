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
# Compile a program and extract the number of LLVM instructions.
#

import os
import sys
import glob
import subprocess

from absl import app, flags, logging

from yacos.essential import IO
from yacos.essential import Sequence
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


def generate_bitcode(filename, message=''):
    """Generate bitcode."""
    if filename.endswith('.c'):
        compiler = 'clang'
        new_filename = filename[:-2] + '.bc'
    elif filename.endswith('.cpp'):
        compiler = 'clang++'
        new_filename = filename[:-4] + '.bc'
    else:
        return None

    cmdline = '{} -c -Xclang -disable-O0-optnone -w -emit-llvm {} -o {}'.format(compiler,
                                                                                filename,
                                                                                new_filename)

    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        return None # sys.exit(1)

    return new_filename


def optimize(filename, sequence, message=''):
    """Optimize."""
    new_filename = '{}_opt'.format(filename)
    cmdline = 'opt {} {} -o {}'.format(Sequence.name_pass_to_string(sequence),
                                       filename,
                                       new_filename)

    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        return None # sys.exit(1)

    return new_filename


def remove_temp_file(filename, message=''):
    """Remove a temporary file."""
    cmdline = 'rm -f {}'.format(filename)
    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        sys.exit(1)


def remove_bc_file(folder, message=''):
    """Remove bc files."""
    cmdline = 'rm -f {}/*.bc'.format(folder)
    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        sys.exit(1)


def find_source_name(source, folder=''):
    """Extract the filename."""
    if folder:
        name = source.replace('{}/'.format(folder), '')
    idx = name.rfind('.')
    return name[:idx]


def execute(argv):
    """Execute."""
    del argv

    FLAGS = flags.FLAGS

    # Verify datset directory.
    if not os.path.isdir(FLAGS.dataset_directory):
        logging.error('Dataset directory {} does not exist.'.format(
            FLAGS.dataset_directory)
        )
        sys.exit(1)

    # Load Sequences file.
    sequences = IO.load_yaml_or_fail(FLAGS.sequences_filename)

    folders = [
                os.path.join(FLAGS.dataset_directory, subdir)
                for subdir in os.listdir(FLAGS.dataset_directory)
                if os.path.isdir(os.path.join(FLAGS.dataset_directory, subdir))
              ]

    # Instantiate the LLVM Driver
    driver = LLVMDriver()

    # Load data from all folders
    nof_instructions = {}
    for folder in folders:

        if not(folder.endswith('C') or folder.endswith('C++')):
            continue

        sources = glob.glob('{}/*'.format(folder))

        if len(sources) < 3000:
            continue

        folder_name = folder.replace('{}/'.format(FLAGS.dataset_directory), '')
        nof_instructions[folder_name] = {}

        for source in sources:

            source_name = find_source_name(source, folder)

            if not source_name:
                continue

            filename = generate_bitcode(source)

            if not filename:
                continue

            nof_instructions[folder_name][source_name] = {}

            for sequence_key, sequence in sequences.items():

                if FLAGS.opt:
                    opt_filename = optimize(filename, sequence['seq'])
                else:
                    driver.setOptimizations(sequence['seq'])
                    opt_filename = filename

                # Instantiate the builder.
                builder = R.LLVMInstsBuilder(driver)

                try:
                    extractionInfo = builder.ir_to_info(opt_filename)
                    insts = [data.instructions for data in extractionInfo.functionInfos]
                except Exception:
                    if FLAGS.opt:
                        remove_temp_file(opt_filename)
                    continue

                nof_instructions[folder_name][source_name][sequence_key] = sum(insts)

                if FLAGS.opt:
                    remove_temp_file(opt_filename)

            remove_bc_file(folder)

    IO.dump_yaml(nof_instructions, FLAGS.output_filename)


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_string('output_filename',
                        'nof_instructions.yaml',
                        'Output filename')
    flags.DEFINE_string('sequences_filename',
                        None,
                        'Sequences filename')
    flags.DEFINE_boolean('opt',
                         True,
                         'Use Opt to compile the program')
    flags.mark_flag_as_required('dataset_directory')
    flags.mark_flag_as_required('sequences_filename')

    app.run(execute)
