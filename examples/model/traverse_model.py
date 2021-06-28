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
# Extract functions name from a program.
#

import os
import sys

from absl import app, flags, logging
from networkx.algorithms import traversal
from tqdm import tqdm

from yacos.info import compy as R
from yacos.info.ncc import Inst2Vec

from yacos.essential import IO, similarity
from yacos.essential import Goal
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.essential import Similarity

from yacos.info.compy.extractors import LLVMDriver

from yacos.model import RepresentationExtractor
from yacos.model import GraphFromSequences




def create_sim_table(train,
                     test,
                     distances):
    """
        create a table of similarity between the train program
        and test programs.
        Parameters
        ----------
        train: list
            list of train programs names
        test: list
            list of test programs names
        distances: matrix 
            matrix of distances where distances[i][j]
            is the distance of train[i] to test[j]

        Returns
        -------
        A dict named table where table[test[j]][train[i]]
        have the value of distance between test[j] and train[i] 
    """
    table = {}
    for j in range(len(test)):
        prg_test = test[j]
        table[prg_test] = {}
        for i in range(len(train)):
            prg_train = train[i]
            table[prg_test][prg_train] = distances[i][j]
    return table

def most_similar(train,
                 test,
                 distances):
    """
        get the most similar program name
        Parameters
        ----------
        train: list
            a list of string containing names of training programs
        test: list
            a list containing names of test programs
        distances: matrix 
            matrix of distances where distances[i][j]
            is the distance of train[i] to test[j]
        
        Return
        ------
        a list bench_list where bench_list[i] is the name of the
        closest program from train of test[i]
    """

    bench_list = {}#[None for i in range(len(test))]
    for j in range(len(test)):
        bench = train[0]
        dist = distances[0][j]
        for i in range(len(train)):
            #print(train[i],test[j],distances[i][j])
            if distances[i][j] < dist:
                bench = train[i]
                dist = distances[i][j]
        bench_list[test[j]] = bench

    return bench_list


def execute(argv):
    """Generate random sequences for each benchmark."""
    del argv

    FLAGS = flags.FLAGS

    # Verify the benchmark file
    if not os.path.isfile(FLAGS.benchmarks_filename):
        logging.error('File {} does not exist.'.format(
            FLAGS.benchmarks_filename)
        )
        sys.exit(1)


    # Verify benchmark directory
    if not os.path.isdir(FLAGS.benchmarks_directory):
        logging.error('Benchmarks directory {} does not exist.'.format(
            FLAGS.benchmarks_directory)
        )
        sys.exit(1)
    
    #Verify graph file 
    graph_fullpath = os.path.join(FLAGS.model_directory,
                                  FLAGS.graph_filename)
    if not os.path.isfile(graph_fullpath):
        logging.error('File {} does not exist.'.format(
            graph_fullpath)
        )
    #verify representation file
    representation_fullpath = os.path.join(FLAGS.model_directory,
                                           FLAGS.representation_filename)
    
    if not os.path.isfile(representation_fullpath):
        logging.error('File {} does not exist.'.format(
            representation_fullpath)
        )

    #check the representation type and set the function that will be extract it
    if FLAGS.representation_type == 'inst2vec':
        get_repr_function = RepresentationExtractor.get_inst2vec_features
    else:
        print('invalid representation')
        sys.exit(1)

    Model = GraphFromSequences(graph_fullpath,
                               representation_fullpath)
    #Load model
    Model.load()
    
    #get representations of train programs
    train_representation = Model.get_representations()




    #extract representation of all test benchmakrs
    test_benchmarks = IO.load_yaml(FLAGS.benchmarks_filename)
    test_representation = {}
    for bench in test_benchmarks:
        idx = bench.find('.')
        bench_collection = bench[:idx]
        bench_name = bench[idx+1:]
        bench_path = os.path.join(FLAGS.benchmarks_directory,
                                  bench_collection,
                                  bench_name)
        bench_representation = get_repr_function(bench_path)
        test_representation[bench_name] = bench_representation
    
    #calculate distances beteween train and test
    test, train, dist = Similarity.euclidean_distance_from_data(test_representation,
                                                                train_representation)
    

    inital_progs = most_similar(train, test, dist)
    if FLAGS.traversal_algorithm in ['similar', 'weighted', 'backtracking']:
        similarity_table = create_sim_table(train, test, dist)

    goal = Goal.prepare_goal(FLAGS.goals,FLAGS.weights)

    #for i in range(len(test_benchmarks)):
    #    bench = test_benchmarks[i]
    for bench in test_benchmarks:
        out_dict = {}
        idx = bench.find('.')
        bench_collection = bench[:idx]
        bench_name = bench[idx+1:]
        ini_train_bench = inital_progs[bench_name]
        bench_path = os.path.join(FLAGS.benchmarks_directory,
                                  bench_collection,
                                  bench_name)
        out_dict['sequences'] = {}
        out_dict['sequences'][FLAGS.traversal_algorithm] = []
        ini = Model.get_initial_opt(ini_train_bench)
        if FLAGS.traversal_algorithm == 'cost':            
            result = Model.traversal_cost(ini,
                                          FLAGS.sequence_length)
        elif FLAGS.traversal_algorithm == 'similar':
            result = Model.traversal_similar(ini,
                                             FLAGS.sequence_length,
                                             similarity_table[bench_name])
        elif FLAGS.traversal_algorithm == 'weighted':
            result = Model.traversal_weighted(ini,
                                              FLAGS.sequence_length,
                                              similarity_table[bench_name])
        elif FLAGS.traversal_algorithm == 'backtracking':
            result = Model.traversal_backtracking(ini,
                                                  FLAGS.sequence_length,
                                                  similarity_table[bench_name],
                                                  bench_path,
                                                  goal,
                                                  get_compile_time=True)
        else:
            logging.error('Invalid traversal algorithm {}'.format(
                FLAGS.traversal_algorithm)
            )
            sys.exit(1)
        s = result['sequence']
        str_seq = ' '.join(s)
        g = Engine.evaluate(goal,
                            str_seq,
                            'opt', 
                            bench_path)
        out_dict['sequences'][FLAGS.traversal_algorithm].append(
            {'sequence': s,
             'goal': g,
             'focused_length': FLAGS.sequence_length}
        )
        os.makedirs(FLAGS.output_directory, exist_ok=True)
        out_file = os.path.join(FLAGS.output_directory,bench)
        out_file = out_file + '.yaml'
        IO.dump_yaml(out_dict,out_file)

# Execute
if __name__ == '__main__':
    flags.DEFINE_string('benchmarks_filename',
                        None,
                        'YAML Benchmarks Files')
    flags.DEFINE_string('benchmarks_directory',
                        None,
                        'Benchmarks directory')
    flags.DEFINE_string('model_directory',
                        None,
                        'Directory that the model is stored')
    flags.DEFINE_string('output_directory',
                         None,
                         'dierctory to store results')
    flags.DEFINE_string('graph_filename',
                        'model_graph.yaml',
                        'Filename that the graph of the model will be stored ')
    flags.DEFINE_string('representation_type',
                        'inst2vec',
                        'Representation used to benchmarks ')
    flags.DEFINE_string('representation_filename',
                        'model_representation.pck',
                        'Filename that the programs representation will be stored ')
    flags.DEFINE_string('traversal_algorithm',
                        'cost',
                        'Traversal Algorithm (cost|similar|weighted|backtracking)')
    flags.DEFINE_integer('sequence_length',
                         30,
                         'The length that the algorithm will try to build the sequence')
    flags.DEFINE_spaceseplist('goals',
                              ['code_size'],
                              'List of goals (separed by spaces)')
    flags.DEFINE_spaceseplist('weights',
                              [1],
                              'List of the weights of each goal (separed by spaces)')
    

    flags.mark_flag_as_required('benchmarks_filename')
    flags.mark_flag_as_required('benchmarks_directory')
    flags.mark_flag_as_required('model_directory')
    flags.mark_flag_as_required('output_directory')

    app.run(execute)
