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
import glob
import numpy as np
import pandas as pd

from absl import app, flags, logging

from yacos.essential import IO
from yacos.info import compy as R
from yacos.info.compy.extractors import ClangDriver, LLVMDriver

from sklearn import model_selection
import matplotlib.pyplot as plt

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarDiGraph

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


def graph2stellar(data, n_features, graph='ir'):
    """Convert the graph to StellarGraph representation."""

    s_labels = []
    s_graphs = []
    for label, graphs in data.items():
        for graph in graphs:
            if n_features == 'inst2vec':
                nodes_features = graph.get_nodes_inst2vec_embeddings()
            elif n_features == 'ir2vec':
                nodes_features = graph.get_nodes_ir2vec_embeddings()
            elif n_features == 'word2vec_ast':
                nodes_features = graph.get_nodes_word2vec_embeddings('ast')
            elif n_features == 'word2vec_ir':
                nodes_features = graph.get_nodes_word2vec_embeddings('ir')
            elif n_features == 'word2vec_asm':
                nodes_features = graph.get_nodes_word2vec_embeddings('asm')
            elif n_features == 'bag_of_word':
                nodes_features = graph.get_nodes_bag_of_words_embeddings(graph)

            n_index = [index for index, _, _ in nodes_features]
            n_features = [features for _, _, features in nodes_features]

            node_data = pd.DataFrame({"embeddings": n_features}, index=n_index)

            edges = graph.get_edges_dataFrame()

            s_graph = StellarDiGraph(node_data,
                                     edges=edges,
                                     edge_type_column="type")
            # print(s_graph.info(show_attributes=True, truncate=None))
            s_labels.append(label)
            s_graphs.append(s_graph)

    return s_graphs, pd.Series(s_labels, name='label', dtype="category")


def prepare_data(data_directory,
                 graph_type):
    """Extract the representation from the source code."""

    if graph_type in ['ast', 'astdata', 'astdatacfg']:
        # Instantiate the Clang driver.
        driver = ClangDriver(
            ClangDriver.ProgrammingLanguage.C,
            ClangDriver.OptimizationLevel.O0,
            [],
            ["-xcl"]
        )
        # Define the builder
        builder = R.ASTGraphBuilder(driver)
    else:
        # Instantiate the LLVM driver.
        driver = LLVMDriver()
        # Define the builder
        builder = R.LLVMGraphBuilder(driver)

    # Define the visitor
    visitors = {
                # Clang
                'ast': R.ASTVisitor,
                'astdata': R.ASTDataVisitor,
                'astdatacfg': R.ASTDataCFGVisitor,
                # LLVM
                'programl': R.LLVMProGraMLVisitor,
                'programlnoroot': R.LLVMProGraMLNoRootVisitor,
                'cfg': R.LLVMCFGVisitor,
                'cfgcompact': R.LLVMCFGCompactVisitor,
                'cfgcall': R.LLVMCFGCallVisitor,
                'cfgcallnoroot': R.LLVMCFGCallNoRootVisitor,
                'cfgcallcompact': R.LLVMCFGCallCompactVisitor,
                'cfgcallcompact1e': R.LLVMCFGCallCompactOneEdgeVisitor,
                'cfgcallcompactnoroot': R.LLVMCFGCallCompactNoRootVisitor,
                'cfgcallcompact1enoroot': R.LLVMCFGCallCompactOneEdgeNoRootVisitor,
                'cdfg': R.LLVMCDFGVisitor,
                'cdfgcompact': R.LLVMCDFGCompactVisitor,
                'cdfgcompact1e': R.LLVMCDFGCompactOneEdgeVisitor,
                'cdfgcall': R.LLVMCDFGCallVisitor,
                'cdfgcallnoroot': R.LLVMCDFGCallNoRootVisitor,
                'cdfgcallcompact': R.LLVMCDFGCallCompactVisitor,
                'cdfgcallcompact1e': R.LLVMCDFGCallCompactOneEdgeVisitor,
                'cdfgcallcompactnoroot': R.LLVMCDFGCallCompactNoRootVisitor,
                'cdfgcallcompact1enoroot': R.LLVMCDFGCallCompactOneEdgeNoRootVisitor,
                'cdfgplus': R.LLVMCDFGPlusVisitor,
                'cdfgplusnoroot': R.LLVMCDFGPlusNoRootVisitor
                }

    folders = [
                os.path.join(data_directory, subdir)
                for subdir in os.listdir(data_directory)
                if os.path.isdir(os.path.join(data_directory, subdir))
              ]

    representation = {}
    # Load data from all folders
    for folder in folders:
        label = folder.replace('{}/'.format(data_directory), '')
        sources = glob.glob('{}/*.ll'.format(folder))
        representation[label] = []
        for source in sources:
            # Extract "information" from the file
            # (data to construct the graph).
            if graph_type == 'asmcompact':
                print('not implemented yet')
                sys.exit(1)
            else:
                extractionInfo = builder.ir_to_info(source)
                # Build the graph.
                graph = builder.info_to_representation(extractionInfo,
                                                       visitors[graph_type])
            representation[label].append(graph)
    return representation


def execute(argv):
    """Extract a graph representation."""
    del argv

    # Print summary
    print('='*80, flush=True)
    print('Classify applications into 104 classes given their raw code.')
    print('='*80, flush=True)
    FLAGS = flags.FLAGS
    print('Deep Graph Convolutional Neural Network')
    print('='*80, flush=True)

    # Verify datset directory.
    if not os.path.isdir(FLAGS.dataset_directory):
        logging.error('Dataset directory {} does not exist.'.format(
            FLAGS.dataset_directory)
        )
        sys.exit(1)

    #
    # IMPORT THE DATA
    #

    # Prepare the datasets
    dataset = prepare_data(FLAGS.dataset_directory, FLAGS.graph)

    if FLAGS.graph in ['ast', 'astdata', 'astdatacfg']:
        graph = 'ast'
        if FLAGS.node_features == 'word2vec':
            n_features = 'word2vec_ast'
        elif FLAGS.node_features == 'bag_of_words':
            n_features = 'bag_of_words'
        else:
            logging.error('Invalid node features.')
            sys.exit(1)
    elif FLAGS.graph in ['asmcompact']:
        graph = 'asm'
        if FLAGS.node_features == 'word2vec':
            n_features = 'word2vec_asm'
        elif FLAGS.node_features == 'bag_of_words':
            n_features = 'bag_of_words'
        else:
            logging.error('Invalid node features.')
            sys.exit(1)
    else:
        graph = 'ir'
        if FLAGS.node_features == 'word2vec':
            n_features = 'word2vec_ir'
        elif FLAGS.nodes_features == 'inst2vec':
            n_features = 'inst2vec'
        elif FLAGS.node_features == 'ir2vec':
            n_features = 'ir2vec'
        elif FLAGS.node_features == 'bag_of_words':
            n_features = 'bag_of_words'

    graphs, graph_labels = graph2stellar(dataset, n_features, graph)

    # Summary statistics of the sizes of the graphs
    print('Dataset', flush=True)
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=['nodes', 'edges'],
    )
    print('\n', summary.describe().round(1), flush=True)
    print('\n', graph_labels.value_counts().to_frame(), flush=True)
    print('='*80, flush=True)

    # Encode class values
    graph_labels = pd.get_dummies(graph_labels)
    classes = graph_labels.shape[1]

    #
    # PREPARE GRAPH GENERATOR
    #

    generator = PaddedGraphGenerator(graphs=graphs)

    #
    # CREATE THE KERAS GRAPH CLASSIFICATION MODEL
    #

    # First we create the base DGCNN model that includes
    # the graph convolutional and SortPooling layers
    k = 35  # the number of rows for the output tensor
    layer_sizes = [256, 256, 256, classes]

    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["relu", "relu", "relu", "relu"],
        k=k,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    # Add the convolutional, max pooling, and dense layers
    x_out = Conv1D(filters=16,
                   kernel_size=sum(layer_sizes),
                   strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
    x_out = Flatten()(x_out)
    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.5)(x_out)
    outputs = Dense(units=classes, activation="sigmoid")(x_out)

    # Create the model and prepare it for training by specifying
    # the loss and optimisation algorithm.
    model = Model(inputs=x_inp, outputs=outputs)

    model.compile(
        optimizer=Adam(lr=0.0001),
        loss=categorical_crossentropy,
        metrics=["acc"]
    )

    #
    # TRAIN THE MODEL
    #

    # Split the dataset to training, validate and test sets
    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels,
        train_size=1.0-FLAGS.test_ratio,
        test_size=FLAGS.test_ratio,
        stratify=graph_labels
    )

    train_graphs, val_graphs = model_selection.train_test_split(
        train_graphs,
        train_size=FLAGS.train_ratio,
        test_size=FLAGS.val_ratio,
        stratify=train_graphs
    )

    print('Training:', train_graphs.shape[0], flush=True)
    print('Validation:', val_graphs.shape[0], flush=True)
    print('Test:', test_graphs.shape[0], flush=True)
    print('='*80, flush=True)

    # Prepares the data for training
    gen = PaddedGraphGenerator(graphs=graphs)

    train_gen = gen.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=50,
        symmetric_normalization=False,
    )

    val_gen = gen.flow(
        list(val_graphs.index - 1),
        targets=val_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )

    # Train the model
    verbose = 1 if FLAGS.verbose else 0
    history = model.fit(
        train_gen,
        epochs=FLAGS.epochs,
        verbose=verbose,
        validation_data=val_gen,
        shuffle=True
    )

    #
    # EVALUATE THE MODEL
    #

    # Calculate the performance of the trained model on the test data.
    test_metrics = model.evaluate(test_gen)
    if verbose:
        print('='*80, flush=True)

    print('Test Set Metrics', flush=True)
    print('='*80)
    test_metrics_dict = {}
    for name, val in zip(model.metrics_names, test_metrics):
        print('{}: {:0.4f}'.format(name, val), flush=True)
        test_metrics_dict[name] = val

    #
    # PREDICT
    #

    predicted = model.predict(test_gen)

    #
    # STORE THE RESULTS
    #

    # Create the output directory.
    os.makedirs(FLAGS.results_directory, exist_ok=True)

    # Save the history
    IO.dump_yaml(history.history,
                 '{}/history.yaml'.format(FLAGS.results_directory))

    # Save the summary
    IO.dump_yaml(summary.describe().to_dict(),
                 '{}/summary.yaml'.format(FLAGS.results_directory))

    # Save the test metrics
    IO.dump_yaml(test_metrics_dict,
                 '{}/test_metrics.yaml'.format(FLAGS.results_directory))

    # Save the graphs (labels)
    train_graphs.to_pickle('{}/train_graphs.pkl'.format(FLAGS.results_directory))

    val_graphs.to_pickle('{}/val_graphs.pkl'.format(FLAGS.results_directory))

    test_graphs.to_pickle('{}/test_graphs.pkl'.format(FLAGS.results_directory))

    # Save the prediction
    np.savez_compressed('{}/predict.npz'.format(FLAGS.results_directory),
                        encoded=graph_labels[test_graphs.shape[0]:],
                        predicted=predicted)

    # Plot the training history
    # (losses and accuracies for the train and test data).
    figure = sg.utils.plot_history(history, return_figure=True)
    plt.figure(figure)
    plt.savefig('{}/history.pdf'.format(FLAGS.results_directory))


# Execute
if __name__ == '__main__':
    # app
    flags.DEFINE_string('results_directory',
                        None,
                        'Results directory')
    flags.DEFINE_string('dataset_directory',
                        None,
                        'Dataset directory')
    flags.DEFINE_float('train_ratio',
                       0.75,
                       'Training ratio')
    flags.DEFINE_float('val_ratio',
                       0.25,
                       'Validation ratio')
    flags.DEFINE_float('test_ratio',
                       0.20,
                       'Test ratio')
    flags.DEFINE_integer('epochs',
                         100,
                         'Epochs')
    flags.DEFINE_enum('graph',
                      'cdfgcallcompactnoroot',
                      [
                        'ast',
                        'astdata',
                        'astdatacfg',
                        'programl',
                        'programlnoroot',
                        'cfg',
                        'cfgcompact',
                        'cfgcall',
                        'cfgcallnoroot',
                        'cfgcallcompact',
                        'cfgcallcompact1e'
                        'cfgcallcompactnoroot',
                        'cfgcallcompact1enoroot',
                        'cdfg',
                        'cdfgcompact',
                        'cdfgcompact1e',
                        'cdfgcall',
                        'cdfgcallnoroot',
                        'cdfgcallcompact',
                        'cdfgcallcompact1e',
                        'cdfgcallcompactnoroot',
                        'cdfgcallcompact1enoroot',
                        'cdfgplus',
                        'cdfgplusnoroot',
                        'asmcompact'
                      ],
                      'The type of the graph')
    flags.DEFINE_enum('node_features',
                      'bag_of_word',
                      [
                        'bag_of_words',
                        'inst2vec',
                        'word2vec',
                        'ir2vec'
                      ],
                      'Node embeddings is word2vec')
    flags.DEFINE_boolean('verbose',
                         False,
                         'Verbose')

    flags.mark_flag_as_required('dataset_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
