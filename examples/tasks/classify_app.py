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
from yacos.info.compy.extractors import LLVMDriver

import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarDiGraph

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


def get_num_labels(indexes,
                   labels):
    """Get the number of labels"""
    num_labels = {}
    for index in indexes:
        if labels[index] not in num_labels:
            num_labels[labels[index]] = 0
        num_labels[labels[index]] += 1
    return num_labels


def graph2stellar(graph):
    """Convert the graph to StellarGraph representation."""

    nodes_features = graph.get_nodes_inst2vec_embeddings()
    edges = graph.get_edges_dataFrame()

    s_graph = StellarDiGraph(nodes_features,
                             edges=edges,
                             edge_type_column="type")
    # print(s_graph.info(show_attributes=True, truncate=None))
    return s_graph


def prepare_data(data_directory,
                 graph_type,
                 ratio):
    """Extract the representation from the source code."""
    # Instantiate the LLVM driver.
    driver = LLVMDriver()

    # Define the builder
    builder = R.LLVMGraphBuilder(driver)

    # Define the visitor
    visitors = {'programl': R.LLVMProGraMLVisitor,
                'programlnoroot': R.LLVMProGraMLNoRootVisitor,
                'cfg': R.LLVMCFGVisitor,
                'cfgcompact': R.LLVMCFGCompactVisitor,
                'cfgcall': R.LLVMCFGCallVisitor,
                'cfgcallnoroot': R.LLVMCFGCallNoRootVisitor,
                'cfgcallcompact': R.LLVMCFGCallCompactVisitor,
                'cfgcallcompactnoroot': R.LLVMCFGCallCompactNoRootVisitor,
                'cdfg': R.LLVMCDFGVisitor,
                'cdfgcompact': R.LLVMCDFGCompactVisitor,
                'cdfgcall': R.LLVMCDFGCallVisitor,
                'cdfgcallnoroot': R.LLVMCDFGCallNoRootVisitor,
                'cdfgcallcompact': R.LLVMCDFGCallCompactVisitor,
                'cdfgcallcompactnoroot': R.LLVMCDFGCallCompactNoRootVisitor,
                'cdfgplus': R.LLVMCDFGPlusVisitor,
                'cdfgplusnoroot': R.LLVMCDFGPlusNoRootVisitor}

    index = 0
    graphs = []
    labels = []
    indexes = {'ir_train': [], 'ir_val': [], 'ir_test': []}

    # Load data from all folders
    for dir in ['ir_train', 'ir_val', 'ir_test']:
        top_dir = os.path.join(data_directory, dir)
        folders = [
                    os.path.join(top_dir, subdir)
                    for subdir in os.listdir(top_dir)
                    if os.path.isdir(os.path.join(top_dir, subdir))
                  ]

        for folder in folders:
            label = folder.replace('{}/'.format(top_dir), '')
            sources = glob.glob('{}/*.ll'.format(folder))

            nof_dataset_itens = int(len(sources) * ratio)

            for item, source in enumerate(sources):
                # Extract "information" from the file
                # (data to construct the graph).
                extractionInfo = builder.ir_to_info(source)
                # Build the graph.
                graph = builder.info_to_representation(extractionInfo,
                                                       visitors[graph_type])

                indexes[dir].append(index)
                graphs.append(graph2stellar(graph))
                labels.append(label)

                index += 1

                if item == nof_dataset_itens - 1:
                    break

    labels = pd.Series(labels, name='label', dtype="category")

    return graphs, labels, indexes


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
    graphs, graph_labels, graph_indexes = prepare_data(FLAGS.dataset_directory,
                                                       FLAGS.graph,
                                                       FLAGS.ratio)

    # Summary statistics of the sizes of the graphs
    print('Dataset', flush=True)
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=['nodes', 'edges'],
    )

    print('\n', summary.describe().round(1), flush=True)
    print('\n', graph_labels.value_counts().to_frame(), flush=True)
    print('='*80, flush=True)

    # Dataset statistics
    dataset = {'training': {}, 'validation': {}, 'test': {}}

    training = len(graph_indexes['ir_train'])
    dataset['training']['counter'] = training

    validation = len(graph_indexes['ir_val'])
    dataset['validation']['counter'] = validation

    test = len(graph_indexes['ir_test'])
    dataset['test']['counter'] = test

    print('Training:', training, flush=True)
    dataset['training']['labels'] = {}
    num_labels = get_num_labels(graph_indexes['ir_train'], graph_labels)
    for label, counter in num_labels.items():
        print('\t', label, '\t', counter)
        dataset['training']['labels'][label] = counter

    print('Validation:', validation, flush=True)
    dataset['validation']['labels'] = {}
    num_labels = get_num_labels(graph_indexes['ir_val'], graph_labels)
    for label, counter in num_labels.items():
        print('\t', label, '\t', counter)
        dataset['validation']['labels'][label] = counter

    print('Test:', test, flush=True)
    dataset['test']['labels'] = {}
    num_labels = get_num_labels(graph_indexes['ir_test'], graph_labels)
    for label, counter in num_labels.items():
        print('\t', label, '\t', counter)
        dataset['test']['labels'][label] = counter

    print('='*80, flush=True)

    test_labels_original = graph_labels[training+validation:]

    # Encode class values
    graph_labels = pd.get_dummies(graph_labels)
    classes = graph_labels.shape[1]

    test_labels_encoded = graph_labels[training+validation:]

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
    outputs = Dense(units=classes, activation="softmax")(x_out)

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

    # Prepares the data for training
    gen = PaddedGraphGenerator(graphs=graphs)

    train_gen = gen.flow(
        graph_indexes['ir_train'],
        targets=graph_labels[:training],
        batch_size=50,
        symmetric_normalization=False,
    )

    eval_gen = gen.flow(
        graph_indexes['ir_val'],
        targets=graph_labels[training:training+validation],
        batch_size=1,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        graph_indexes['ir_test'],
        targets=graph_labels[training+validation:],
        batch_size=1,
        symmetric_normalization=False,
    )

    # Train the model
    verbose = 1 if FLAGS.verbose else 0
    history = model.fit(
        train_gen,
        epochs=FLAGS.epochs,
        verbose=verbose,
        validation_data=eval_gen,
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

    # Save the dataset
    IO.dump_yaml(dataset,
                 '{}/dataset.yaml'.format(FLAGS.results_directory))

    # Save the metrics
    IO.dump_yaml(test_metrics_dict,
                 '{}/test_metrics.yaml'.format(FLAGS.results_directory))

    # Save the prediction
    np.savez_compressed('{}/predict.npz'.format(FLAGS.results_directory),
                        original=test_labels_original,
                        encoded=test_labels_encoded,
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
    flags.DEFINE_integer('epochs',
                         100,
                         'Epochs')
    flags.DEFINE_float('ratio',
                       1.0,
                       'Use <ratio> of the dataset')
    flags.DEFINE_enum('graph',
                      'cdfgcallcompactnoroot',
                      [
                        'programl',
                        'programlnoroot',
                        'cfg',
                        'cfgcompact',
                        'cfgcall',
                        'cfgcallnoroot',
                        'cfgcallcompact',
                        'cfgcallcompactnoroot',
                        'cdfg',
                        'cdfgcompact',
                        'cdfgcall',
                        'cdfgcallnoroot',
                        'cdfgcallcompact',
                        'cdfgcallcompactnoroot',
                        'cdfgplus',
                        'cdfgplusnoroot'
                      ],
                      'The type of the graph')
    flags.DEFINE_boolean('verbose',
                         False,
                         'Verbose')

    flags.mark_flag_as_required('dataset_directory')
    flags.mark_flag_as_required('results_directory')

    app.run(execute)
