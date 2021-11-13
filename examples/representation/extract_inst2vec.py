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
import numpy as np

from absl import app, flags, logging

from yacos.info.ncc import Inst2Vec


def execute(argv):
    """Extract a graph representation."""
    del argv

    FLAGS = flags.FLAGS

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

    # Extract int2vec
    inst2vec = {}
    max_length = []

    for folder in folders:
        inst2vec[folder] = {}
        # Extract "inst2vec" from the file
        Inst2Vec.prepare_benchmark(folder)
        rep = Inst2Vec.extract(data_type="index")
        for bench, indexes in rep.items():
            inst2vec[folder][bench] = indexes
            max_length.append(len(indexes))
        Inst2Vec.remove_data_directory()

    # Padding
    max_length = max(max_length)
    unk_idx, _ = Inst2Vec.unknown
    embeddings = Inst2Vec.embeddings

    for folder, data in inst2vec.items():
        # Create the output directory.
        outdir = os.path.join(folder.replace(FLAGS.dataset_directory,
                              'inst2vec'))
        os.makedirs(outdir, exist_ok=True)

        for bench, indexes in data.items():
            if FLAGS.index:
                padding = [idx for idx in indexes]
                if FLAGS.padding:
                    for i in range(len(indexes), max_length):
                        padding.append(unk_idx)
            else:
                padding = [list(embeddings[idx]) for idx in indexes]
                if FLAGS.padding:
                    for i in range(len(indexes), max_length):
                        padding += list(embeddings[unk_idx])

            filename = os.path.join(outdir, bench)
            np.savez_compressed(filename, values=padding)

    del embeddings


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_boolean('index',
                         False,
                         'Extract only the indexes.')
    flags.DEFINE_boolean('padding',
                         False,
                         'Padding the representation.')
    flags.mark_flag_as_required('dataset_directory')

    app.run(execute)
