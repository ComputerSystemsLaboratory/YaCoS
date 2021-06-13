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
# Clustering.
#
# This example clusters ir2vec data. So we load the program representation.
# Look: examples/representation/ir2vec_from_ir.py.
#

import os
import sys
import numpy as np
import pandas as pd

from absl import app, flags, logging
from tqdm import tqdm

from yacos.essential import IO
from yacos.data_processing import Clustering


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # The benchmarks
    benchmarks = IO.load_yaml_or_fail(FLAGS.benchmarks_filename)
    if not benchmarks:
        logging.error('There are no benchmarks to process')
        sys.exit(1)

    # Verify data directory
    if not os.path.isdir(FLAGS.data_directory):
        logging.error('Data directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)

    # Prepare the data
    info = {}
    for benchmark in tqdm(benchmarks, desc='Preparing the data'):
        index = benchmark.find('.')
        suite_name = benchmark[:index]
        bench_name = benchmark[index+1:]

        data_dir = os.path.join(FLAGS.data_directory,
                                suite_name)

        if not os.path.isdir(data_dir):
            continue

        filename = '{}/{}.npz'.format(data_dir, bench_name)
        vector = np.load(filename, allow_pickle=True)
        info[bench_name] = vector['values'][-1]

    # Dict -> dataFrame
    data_frame = pd.DataFrame.from_dict(info, orient='index')

    # Clustering
    Clustering.kmeans(n_clusters=FLAGS.n_clusters,
                      n_init=100,
                      max_iter=1000)
    Clustering.fit(data_frame)
    clusters, centroids = Clustering.clusters_and_centroids(data_frame)

    # Clusters size
    size = {}
    for key, items in clusters.items():
        size[key] = len(items)

    # Create the output directory.
    os.makedirs(FLAGS.results_directory, exist_ok=True)

    # Store the results
    results = {'centroids': centroids,
               'clusters': clusters,
               'clusters_size': size}
    filename = '{}/{}.yaml'.format(FLAGS.results_directory,
                                   FLAGS.results_filename)
    IO.dump_yaml(results, filename)


# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'Benchmarks')
    flags.DEFINE_string('data_directory',
                        None,
                        'Data directory')
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('results_filename',
                        None,
                        'Results filename')
    flags.DEFINE_integer('n_clusters',
                         None,
                         'Number of clusters')

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('data_directory')
    flags.mark_flag_as_required('results_directory')
    flags.mark_flag_as_required('results_filename')
    flags.mark_flag_as_required('n_clusters')

    app.run(execute)
