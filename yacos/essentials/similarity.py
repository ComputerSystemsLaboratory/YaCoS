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

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx

from absl import logging as lg
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial import distance_matrix as d_matrix

from yacos.essentials import IO


class Similarity:
    """Static class to measuse the similarity between programs."""

    __version__ = '1.0.0'

    __d2v_model_llvm_seq = None
    __d2v_model_syntax_seq = None
    __d2v_model_syntax_token_kind = None
    __d2v_model_syntax_token_kind_variable = None

    __d2v_dir = 'yacos/doc2vec'

    @staticmethod
    def __populate_data(training_benchmarks,
                        training_directory,
                        test_benchmarks,
                        test_directory):
        """Create test and training data.

        Parameters
        ----------
        training_benchmarks : list

        training_directory : str

        tests_benchmark : list

        test_directory : str

        Returns
        -------
        training_data : pandas.DataFrame

        test_data : pandas.DataFrame
        """
        training_data = {}
        for training_benchmark in training_benchmarks:
            index = training_benchmark.find('.')
            suite_name = training_benchmark[:index]
            bench_name = training_benchmark[index+1:]

            benchmark_dir = os.path.join(training_directory,
                                         suite_name)

            data = IO.load_yaml_or_fail('{}/{}.yaml'.format(benchmark_dir,
                                                            bench_name))
            if data:
                training_data[training_benchmark] = data

        if not training_data:
            lg.error('Training features do not exist.')
            sys.exit(1)

        test_data = {}
        for test_benchmark in test_benchmarks:
            index = test_benchmark.find('.')
            suite_name = test_benchmark[:index]
            bench_name = test_benchmark[index+1:]

            benchmark_dir = os.path.join(test_directory,
                                         suite_name)

            data = IO.load_yaml_or_fail('{}/{}.yaml'.format(benchmark_dir,
                                                            bench_name))
            if data:
                test_data[test_benchmark] = data

        if not test_data:
            lg.error('Training features do not exist.')
            sys.exit(1)

        training_data = pd.DataFrame.from_dict(training_data, orient='index')
        test_data = pd.DataFrame.from_dict(test_data, orient='index')

        return training_data, test_data

    @staticmethod
    def __get_root(g):
        """Find the root node.

        Parameters
        ----------
        g : networkx
        """
        root = None
        for node in g.nodes(data=True):
            if 'root' in node[1]:
                root = node[0]
                break
        else:
            lg.warning('Root node not found (using node 0 as root).')
            return 0
        return root

    @staticmethod
    def __node_match_strong(g1_node, g2_node):
        return g1_node == g2_node

    @staticmethod
    def __node_match_weak(g1_node, g2_node):
        g1_attribute = g1_node['attr'] if 'attr' in g1_node else 'not found'
        g2_attribute = g2_node['attr'] if 'attr' in g2_node else 'not found'
        return g1_attribute == g2_attribute

    @staticmethod
    def __edge_match(g1_edge, g2_edge):
        return g1_edge == g2_edge

    @staticmethod
    def __load_doc2vec_model_syntax_seq():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_seq.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.sys.exit(1)

        Similarity.__d2v_model_syntax_seq = Doc2Vec.load(
                    os.path.join(top_dir, Similarity.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_syntax_token_kind():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.sys.exit(1)

        Similarity.__d2v_model_syntax_token_kind = Doc2Vec.load(
                    os.path.join(top_dir, Similarity.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_syntax_token_kind_variable():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind_variable.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.sys.exit(1)

        Similarity.__d2v_model_syntax_token_kind = Doc2Vec.load(
                    os.path.join(top_dir, Similarity.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_llvm_seq():
        """Load a doc2vec model."""
        MODEL = 'd2v_llvm_seq.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.sys.exit(1)

        Similarity.__d2v_model_syntax_token_kind = Doc2Vec.load(
                    os.path.join(top_dir, Similarity.__d2v_dir, MODEL)
        )

    @staticmethod
    def euclidean_distance_from_data(training_data,
                                     test_data):
        """Euclidean distance.

        Parameters
        ----------
        training_data : dict

        test_data : dict

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data = pd.DataFrame.from_dict(training_data, orient='index')
        test_data = pd.DataFrame.from_dict(test_data, orient='index')
        distance = euclidean_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def euclidean_distance(training_benchmarks,
                           training_directory,
                           test_benchmarks,
                           test_directory):
        """Euclidean distance.

        Parameters
        ----------
        training_benchmarks : list

        training_directory : str

        test_benchmarks : list

        test_directory : list

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data, test_data = Similarity.__populate_data(
            training_benchmarks,
            training_directory,
            test_benchmarks,
            test_directory
        )
        distance = euclidean_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def cosine_distance_from_data(training_data,
                                  test_data):
        """Cosine distance.

        Parameters
        ----------
        training_data : dict

        test_data : dict

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data = pd.DataFrame.from_dict(training_data, orient='index')
        test_data = pd.DataFrame.from_dict(test_data, orient='index')
        distance = cosine_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def cosine_distance(training_benchmarks,
                        training_directory,
                        test_benchmarks,
                        test_directory):
        """Cosine distance.

        Parameters
        ----------
        training_benchmarks : list

        training_directory : str

        test_benchmarks : list

        test_directory : str

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data, test_data = Similarity.__populate_data(
            training_benchmarks,
            training_directory,
            test_benchmarks,
            test_directory
        )
        distance = cosine_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def manhattan_distance_from_data(training_data,
                                     test_data):
        """Manhattan distance.

        Parameters
        ----------
        training_data : dict

        test_data : dict

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data = pd.DataFrame.from_dict(training_data, orient='index')
        test_data = pd.DataFrame.from_dict(test_data, orient='index')
        distance = manhattan_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def manhattan_distance(training_benchmarks,
                           training_directory,
                           test_benchmarks,
                           test_directory):
        """Manhattan distance.

        Parameters
        ----------
        training_benchmarks : list

        training_directory : str

        test_benchmarks : list

        test_directory : str

        Returns
        -------
        training_data : list (rows)

        test_data : list (rows)

        distance : dict
        """
        training_data, test_data = Similarity.__populate_data(
            training_benchmarks,
            training_directory,
            test_benchmarks,
            test_directory
        )
        distance = manhattan_distances(test_data, training_data)
        return training_data.index, test_data.index, distance

    @staticmethod
    def euclidean_distance_from_matrix(A, B, squared=False):
        """Compute all pairwise distances between vectors in A and B.

        Parameters
        ----------
        A : np.array
            shape should be (M, K)

        B : np.array
            shape should be (N, K)

        Return
        ------
        D : np.array
            A matrix D of shape (M, N).  Each entry in D i,j represnets the
            distance between row i in A and row j in B.

        See also
        --------
        A more generalized version of the distance matrix is available from
        scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
        which also gives a choice for p-norm.
        """
        M = A.shape[0]
        N = B.shape[0]

        assert A.shape[1] == B.shape[1], f"The number of components for \
            vectors in A {A.shape[1]} does not match that of B {B.shape[1]}!"

        A_dots = (A*A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
        B_dots = (B*B).sum(axis=1)*np.ones(shape=(M, 1))
        D_squared = A_dots + B_dots - 2 * A.dot(B.T)

        if squared is False:
            zero_mask = np.less(D_squared, 0.0)
            D_squared[zero_mask] = 0.0
            return np.sqrt(D_squared)

        return D_squared

    @staticmethod
    def distance_matrix(A, B, p=2, threshold=1000000):
        """Compute all pairwise distances between vectors in A and B.

        Parameters
        ----------
        A : np.array
            shape should be (M, K)
        B : np.array
            shape should be (N, K)

        p : float, 1 <= p <= infinity
            Which Minkowski p-norm to use.

        threshold : int
            If M * N * K > threshold, algorithm uses a Python loop
                instead of large temporary arrays.

        Return
        ------
        D : np.array
            A matrix D of shape (M, N).  Each entry in D i,j represnets the
            distance between row i in A and row j in B.
        """
        assert A.shape[1] == B.shape[1], f"The number of components for \
            vectors in A {A.shape[1]} does not match that of B {B.shape[1]}!"

        return d_matrix(A, B, p, threshold)

    @staticmethod
    def edit_distance_weak(g1, g2):
        """Edit distance between 2 graphs.

        Parameters
        ----------
        g1 : networkx

        g2 : networkx
        """
        g1_root = Similarity.__get_root(g1)
        g2_root = Similarity.__get_root(g2)

        distance = nx.graph_edit_distance(g1, g2, roots=(g1_root, g2_root))

        return distance

    @staticmethod
    def edit_distance_strong(g1, g2):
        """Edit distance between 2 graphs.

        Parameters
        ----------
        g1 : networkx

        g2 : networkx
        """
        g1_root = Similarity.__get_root(g1)
        g2_root = Similarity.__get_root(g2)

        distance = nx.graph_edit_distance(
                        g1,
                        g2,
                        roots=(g1_root, g2_root),
                        node_match=Similarity.__node_match_strong,
                        edge_match=Similarity.__edge_match,
        )

        return distance

    @staticmethod
    def edit_distance_node_strong(g1, g2):
        """Edit distance between 2 graphs.

        Parameters
        ----------
        g1 : networkx

        g2 : networkx
        """
        g1_root = Similarity.__get_root(g1)
        g2_root = Similarity.__get_root(g2)

        distance = nx.graph_edit_distance(
                        g1,
                        g2,
                        roots=(g1_root, g2_root),
                        node_match=Similarity.__node_match_strong
        )

        return distance

    @staticmethod
    def edit_distance_node_weak(g1, g2):
        """Edit distance between 2 graphs.

        Parameters
        ----------
        g1 : networkx

        g2 : networkx
        """
        g1_root = Similarity.__get_root(g1)
        g2_root = Similarity.__get_root(g2)

        distance = nx.graph_edit_distance(
                        g1,
                        g2,
                        roots=(g1_root, g2_root),
                        node_match=Similarity.__node_match_weak
        )

        return distance

    @staticmethod
    def edit_distance_edge_strong(g1, g2):
        """Edit distance between 2 graphs.

        Parameters
        ----------
        g1 : networkx

        g2 : networkx
        """
        g1_root = Similarity.__get_root(g1)
        g2_root = Similarity.__get_root(g2)

        distance = nx.graph_edit_distance(
                        g1,
                        g2,
                        roots=(g1_root, g2_root),
                        edge_match=Similarity.__edge_match
        )

        return distance

    @staticmethod
    def cosine_similarity_syntax_seq(s1, s2):
        """Cosine similarity between 2 sequences."""
        if not Similarity.__d2v_model_syntax_seq:
            Similarity.__load_doc2vec_model_syntax_seq()
        sim = Similarity.__d2v_model_syntax_seq.docvecs.similarity_unseen_docs(
                    Similarity.__d2v_model_syntax_seq,
                    s1,
                    s2
        )
        return sim

    @staticmethod
    def cosine_similarity_syntax_token_kind(s1, s2):
        """Cosine similarity between 2 sequences."""
        if not Similarity.__d2v_model_syntax_token_kind:
            Similarity.__load_doc2vec_model_syntax_token_kind()
        sim = Similarity.__d2v_model_syntax_token_kind.docvecs.similarity_unseen_docs(
                    Similarity.__d2v_model_syntax_token_kind,
                    s1,
                    s2
        )
        return sim

    @staticmethod
    def cosine_similarity_syntax_token_kind_variable(s1, s2):
        """Cosine similarity between 2 sequences."""
        if not Similarity.__d2v_model_syntax_token_kind_variable:
            Similarity.__load_doc2vec_model_syntax_token_kind_variable()
        sim = Similarity.__d2v_model_syntax_token_kind_variable.docvecs.similarity_unseen_docs(
                    Similarity.__d2v_model_syntax_token_kind_variable,
                    s1,
                    s2
        )
        return sim

    @staticmethod
    def cosine_similarity_llvm_seq(s1, s2):
        """Cosine similarity between 2 sequences."""
        if not Similarity.__d2v_model_llvm_seq:
            Similarity.__load_doc2vec_model_llvm_seq()
        sim = Similarity.__d2v_model_llvm_seq.docvecs.similarity_unseen_docs(
                    Similarity.__d2v_model_llvm_seq,
                    s1,
                    s2
        )
        return sim
