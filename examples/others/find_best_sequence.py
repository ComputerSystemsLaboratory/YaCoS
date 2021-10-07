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
# Find the best sequence
#
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


def optimize(filename, sequence, message=''):
    """Run a command."""
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
        sys.exit(1)

    return new_filename


def remove_temp_file(filename, message=''):
    """Run a command."""
    cmdline = 'rm {}'.format(filename)
    try:
        subprocess.run(cmdline,
                       shell=True,
                       check=True,
                       capture_output=False)
    except subprocess.CalledProcessError:
        if message:
            logging.error(message)
        sys.exit(1)


def execute(argv):
    """Find the best sequence."""
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

    # Load data from all folders
    nof_instructions = {}
    for folder in folders:
        # Extract "ir2vec" from the file
        sources = glob.glob('{}/*.ll'.format(folder))

        folder_name = folder.replace('{}/'.format(FLAGS.dataset_directory), '')
        nof_instructions[folder_name] = {}

        for source in sources:

            name = source.replace('{}/'.format(FLAGS.dataset_directory), '')
            source_name = name[name.find('/')+1:].replace('.ll', '')
            nof_instructions[folder_name][source_name] = {}

            for sequence_key, sequence in sequences.items():
                # Instantiate the LLVM Driver
                driver = LLVMDriver()

                if FLAGS.opt:
                    filename = optimize(source, sequence['seq'])
                else:
                    driver.setOptimizations(sequence['seq'])
                    filename = source

                # Instantiate the builder.
                builder = R.LLVMInstsBuilder(driver)

                try:
                    extractionInfo = builder.ir_to_info(filename)
                    insts = [data.instructions for data in extractionInfo.functionInfos]
                except Exception:
                    if FLAGS.opt:
                        remove_temp_file(filename)
                    continue

                nof_instructions[folder_name][source_name][sequence_key] = sum(insts)

                if FLAGS.opt:
                    remove_temp_file(filename)

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
                         False,
                         'Use Opt to compile the program')
    flags.mark_flag_as_required('dataset_directory')
    flags.mark_flag_as_required('sequences_filename')

    app.run(execute)
