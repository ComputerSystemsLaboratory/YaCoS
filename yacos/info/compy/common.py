"""
Copyright 2020 Alexander Brauckmann.

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
import numpy as np
import pandas as pd
import collections

import networkx as nx
import pygraphviz as pgv

from absl import logging as lg
from gensim.models import Word2Vec, Doc2Vec
from scipy.sparse import csc_matrix
from yacos.essentials import IO
from yacos.info.ncc.inst2vec import inst2vec_preprocess as i2v_pre


class RepresentationBuilder(object):
    """Build a representation."""

    # The tokens in the representation
    _tokens = None

    def __init__(self):
        """Initialize the RepresentationBuilder."""
        self._tokens = collections.OrderedDict()

    def num_tokens(self):
        """Return the number of tokens."""
        return len(self._tokens)

    def get_tokens(self):
        """Return the token."""
        return list(self._tokens.keys())

    def print_tokens(self):
        """Print the tokens."""
        print("-" * 50)
        print("{:<8} {:<25} {:<10}".format("NodeID", "Label", "Number"))
        t_view = [(v, k) for k, v in self._tokens.items()]
        t_view = sorted(t_view, key=lambda x: x[0], reverse=True)
        for v, k in t_view:
            idx = list(self._tokens.keys()).index(k)
            print("{:<8} {:<25} {:<10}".format(str(idx), str(k), str(v)))
        print("-" * 50)


class Sequence(object):
    """Sequence representation."""

    __w2v_model_llvm_seq = None
    __w2v_model_syntax_seq = None
    __w2v_model_syntax_token_kind = None
    __w2v_model_syntax_token_kind_variable = None
    __d2v_model_llvm_seq = None
    __d2v_model_syntax_seq = None
    __d2v_model_syntax_token_kind = None
    __d2v_model_syntax_token_kind_variable = None

    __w2v_dir = 'yacos/info/compy/data/word2vec'
    __d2v_dir = 'yacos/info/compy/data/doc2vec'

    def __init__(self, S, token_types):
        """Initialize a Sequence representation."""
        self.S = S
        self.__token_types = token_types

    def __load_word2vec_model_syntax_seq(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_syntax_seq_skip_gram.model'
        else:
            MODEL = 'w2v_syntax_seq_cbow.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__w2v_model_syntax_seq = Word2Vec.load(
                    os.path.join(top_dir, Sequence.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_syntax_token_kind(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_syntax_token_kind_skip_gram.model'
        else:
            MODEL = 'w2v_syntax_token_kind_cbow.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__w2v_model_syntax_token_kind = Word2Vec.load(
                    os.path.join(top_dir,  Sequence.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_syntax_token_kind_variable(self,
                                                         skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_syntax_token_kind_variable_skip_gram.model'
        else:
            MODEL = 'w2v_syntax_token_kind_variable_cbow.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__w2v_model_syntax_token_kind_variable = Word2Vec.load(
                    os.path.join(top_dir, Sequence.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_llvm_seq(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_llvm_seq_skip_gram.model'
        else:
            MODEL = 'w2v_llvm_seq_cbow.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__w2v_model_llvm_seq = Word2Vec.load(
                    os.path.join(top_dir, Sequence.__w2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_syntax_seq():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_seq.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__d2v_model_syntax_seq = Doc2Vec.load(
                    os.path.join(top_dir, Sequence.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_syntax_token_kind():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__d2v_model_syntax_token_kind = Doc2Vec.load(
                    os.path.join(top_dir, Sequence.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_syntax_token_kind_variable():
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind_variable.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__d2v_model_syntax_token_kind_variable = Doc2Vec.load(
                    os.path.join(top_dir, Sequence.__d2v_dir, MODEL)
        )

    @staticmethod
    def __load_doc2vec_model_llvm_seq():
        """Load a doc2vec model."""
        MODEL = 'd2v_llvm_seq.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Sequence.__d2v_model_llvm_seq = Doc2Vec.load(
                    os.path.join(top_dir, Sequence.__d2v_dir, MODEL)
        )

    def get_token_list(self):
        """Return the list of tokens."""
        node_ints = [self.__token_types.index(tok_str) for tok_str in self.S]
        return node_ints

    def get_token_name_list(self):
        """Return the list of tokens."""
        node_names = [tok_str for tok_str in self.S]
        return node_names

    def get_syntax_seq_word2vec_list(self, skip_gram=False):
        """Return the word2vec for each seq."""
        if not Sequence.__w2v_model_syntax_seq:
            self.__load_word2vec_model_syntax_seq(skip_gram)

        unknown = Sequence.__w2v_model_syntax_seq.wv['unknown']
        tokens = [tok_str for tok_str in self.S]
        nodes = []
        for token in tokens:
            if token in Sequence.__w2v_model_syntax_seq.wv.vocab:
                nodes.append(Sequence.__w2v_model_syntax_seq.wv[token])
            else:
                nodes.append(unknown)

        return np.array(nodes), np.array(unknown)

    def get_syntax_token_kind_word2vec_list(self, skip_gram=False):
        """Return the word2vec for each token."""
        if not Sequence.__w2v_model_syntax_token_kind:
            self.__load_word2vec_model_syntax_token_kind(skip_gram)

        unknown = Sequence.__w2v_model_syntax_token_kind.wv['unknown']
        tokens = [tok_str for tok_str in self.S]
        nodes = []
        for token in tokens:
            if token in Sequence.__w2v_model_syntax_token_kind.wv.vocab:
                nodes.append(Sequence.__w2v_model_syntax_token_kind.wv[token])
            else:
                nodes.append(unknown)

        return np.array(nodes), np.array(unknown)

    def get_syntax_token_kind_variable_word2vec_list(self, skip_gram=False):
        """Return the word2vec for each token."""
        if not Sequence.__w2v_model_syntax_token_kind_variable:
            self.__load_word2vec_model_syntax_token_kind_variable(skip_gram)

        unknown = Sequence.__w2v_model_syntax_token_kind_variable.wv['unknown']
        tokens = [tok_str for tok_str in self.S]
        nodes = []
        for token in tokens:
            if token in Sequence.__w2v_model_syntax_token_kind_variable.wv.vocab:
                nodes.append(Sequence.__w2v_model_syntax_token_kind_variable.wv[token])
            else:
                nodes.append(unknown)

        return np.array(nodes), np.array(unknown)

    def get_llvm_seq_word2vec_list(self, skip_gram=False):
        """Return the word2vec for each token."""
        if not Sequence.__w2v_model_llvm_seq:
            self.__load_word2vec_model_llvm_seq(skip_gram)

        unknown = Sequence.__w2v_model_llvm_seq.wv['unknown']
        tokens = [tok_str for tok_str in self.S]
        nodes = []
        for token in tokens:
            if token in Sequence.__w2v_model_llvm_seq.wv.vocab:
                nodes.append(Sequence.__w2v_model_llvm_seq.wv[token])
            else:
                nodes.append(unknown)

        return np.array(nodes), np.array(unknown)

    def get_syntax_seq_doc2vec(self):
        """Return the doc2vec features."""
        if not Sequence.__d2v_model_syntax_seq:
            self.__load_doc2vec_model_syntax_seq()

        tokens = [tok_str for tok_str in self.S]
        return Sequence.__d2v_model_syntax_seq.infer_vector(tokens)

    def get_syntax_token_kind_doc2vec(self):
        """Return the doc2vec features."""
        if not Sequence.__d2v_model_syntax_token_kind:
            self.__load_doc2vec_model_syntax_token_kind()

        tokens = [tok_str for tok_str in self.S]
        return Sequence.__d2v_model_syntax_token_kind.infer_vector(tokens)

    def get_syntax_token_kind_variable_doc2vec(self):
        """Return the doc2vec features."""
        if not Sequence.__d2v_model_syntax_token_kind_variable:
            self.__load_doc2vec_model_syntax_token_kind_variable()

        tokens = [tok_str for tok_str in self.S]
        return Sequence.__d2v_model_syntax_token_kind_variable.infer_vector(
                            tokens
                        )

    def get_llvm_seq_doc2vec(self):
        """Return the doc2vec features."""
        if not Sequence.__d2v_model_llvm_seq:
            self.__load_doc2vec_model_llvm_seq()

        tokens = [tok_str for tok_str in self.S]
        return Sequence.__d2v_model_llvm_seq.infer_vector(tokens)

    def size(self):
        """Return the length of the sequence."""
        return len(self.S)

    def draw(self, width=8, limit=30, path=None):
        """Create a dot graph."""
        graphviz_graph = pgv.AGraph(
            directed=True,
            splines=False,
            rankdir="LR",
            nodesep=0.001,
            ranksep=0.4,
            outputorder="edgesfirst",
            fillcolor="white",
        )

        remaining_tokens = None
        for i, token in enumerate(self.S):
            if i == limit:
                remaining_tokens = 5

            if remaining_tokens is not None:
                if remaining_tokens > 0:
                    token = "..."
                    remaining_tokens -= 1
                else:
                    break

            subgraph = graphviz_graph.subgraph(
                name="cluster_%i" % i, label="", color="white"
            )

            if i % width == 0:
                graphviz_graph.add_node(i, label=token, shape="box")
                if i > 0:
                    graphviz_graph.add_edge(
                        i - width, i, color="white", constraint=False
                    )
            else:
                subgraph.add_node(i, label=token, shape="box")

            if i > 0:
                if i % width == 0:
                    graphviz_graph.add_edge(i - 1,
                                            i,
                                            constraint=False,
                                            color="gray")
                else:
                    graphviz_graph.add_edge(i - 1, i)

        graphviz_graph.layout("dot")

        return graphviz_graph.draw(path)


class Graph(object):
    """Graph representation."""

    __w2v_model_ast = None
    __w2v_model_ir_graph = None
    __inst2vec_dictionary = None
    __inst2vec_embeddings = None

    __w2v_dir = 'yacos/info/compy/data/word2vec'
    __i2v_dir = 'yacos/info/ncc/data'

    def __init__(self, graph, node_types, edge_types):
        """Initialize a Graph representation."""
        self.G = graph
        self.__node_types = node_types
        self.__edge_types = edge_types

    def __load_inst2vec_data(self):
        """Load the dictionary and embeddings."""
        DICTIONARY = 'inst2vec_augmented_dictionary.pickle'
        EMBEDDINGS = 'inst2vec_augmented_embeddings.pickle'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        filename = os.path.join(top_dir, self.__i2v_dir, DICTIONARY)
        Graph.__inst2vec_dictionary = IO.load_pickle_or_fail(filename)

        filename = os.path.join(top_dir, self.__i2v_dir, EMBEDDINGS)
        Graph.__inst2vec_embeddings = IO.load_pickle_or_fail(filename)

    def __load_word2vec_model_ast(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_ast_skip_gram.model'
        else:
            MODEL = 'w2v_ast_model.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Graph.__w2v_model_ast = Word2Vec.load(os.path.join(top_dir,
                                                           self.__w2v_dir,
                                                           MODEL))

    def __load_word2vec_model_ir_graph(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_ir_graph_skip_gram.model'
        else:
            MODEL = 'w2v_ir_graph_model.model'

        top_dir = os.environ.get('PYTHONPATH')
        if not top_dir:
            lg.error('PYTHONPATH does not exist.')
            sys.sys.exit(1)

        Graph.__w2v_model_ir_graph = Word2Vec.load(
                        os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __get_node_attr_dict(self):
        """Return the node attributes."""
        return collections.OrderedDict(self.G.nodes(data="attr",
                                                    default="N/A"))

    def get_node_str_list(self):
        """Return the node attributes."""
        node_strs = list(self.__get_node_attr_dict().values())
        return node_strs

    def get_node_list(self):
        """Return the nodes."""
        node_strs = list(self.__get_node_attr_dict().values())
        node_ints = [self.__node_types.index(n_str) for n_str in node_strs]
        return node_ints

    def get_nodes_inst2vec_embeddings(self):
        """Return the nodes embeddings (int2vec)."""
        if not Graph.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        augmented = {'!UNK': 8564,
                     '!IDENTIFIER': 8565,
                     '!IMMEDIATE': 8566,
                     '!MAGIC': 8567,
                     '!BB': 8568}

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            if "value" in data:
                nodes.append(
                    Graph.__inst2vec_embeddings[augmented['!IMMEDIATE']])
            elif "inst" in data:
                if type(data["inst"]) is tuple:
                    inst = "\n".join(data["inst"])
                else:
                    inst = data["inst"]
                preprocessed, _ = i2v_pre.preprocess([[inst]])
                preprocessed = i2v_pre.PreprocessStatement(preprocessed[0][0])
                if preprocessed in Graph.__inst2vec_dictionary:
                    embeddings = Graph.__inst2vec_dictionary[preprocessed]
                    embeddings = Graph.__inst2vec_embeddings[embeddings]
                else:
                    embeddings = Graph.__inst2vec_embeddings[augmented['!UNK']]
                nodes.append(embeddings)
            elif "insts" in data:
                emb = []
                for inst in data['insts']:
                    if type(inst) is tuple:
                        inst_ = "\n".join(inst)
                    else:
                        inst_ = inst
                    preprocessed, _ = i2v_pre.preprocess([[inst_]])
                    preprocessed = i2v_pre.PreprocessStatement(
                                        preprocessed[0][0]
                                   )
                    if preprocessed in Graph.__inst2vec_dictionary:
                        embeddings = Graph.__inst2vec_dictionary[preprocessed]
                        embeddings = Graph.__inst2vec_embeddings[embeddings]
                    else:
                        embeddings = Graph.__inst2vec_embeddings[
                                                        augmented['!UNK']
                                                                ]
                    emb.append(embeddings)
                nodes.append([sum(x) for x in zip(*emb)])
            elif "attr" in data:
                if type(data["attr"]) is tuple:
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]
                if label == 'bb':
                    nodes.append(
                        Graph.__inst2vec_embeddings[augmented['!BB']])
                elif label == 'function':
                    nodes.append(
                        Graph.__inst2vec_embeddings[augmented['!MAGIC']])
                else:
                    nodes.append(
                        Graph.__inst2vec_embeddings[augmented['!IDENTIFIER']])
        return np.array(nodes)

    def get_nodes_word2vec_ast_embeddings(self, skip_gram=False):
        """Return the nodes embeddings (word2vec)."""
        if not Graph.__w2v_model_ast:
            self.__load_word2vec_model(skip_gram)

        unknown = Graph.__w2v_model_ast.wv['unknown']

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            label = data["attr"].replace(' ', '')
            if label in Graph.__w2v_model_ast.wv.vocab:
                nodes.append(Graph.__w2v_model_ast[label])
            else:
                nodes.append(unknown)

        return np.array(nodes)

    def get_nodes_word2vec_ir_graph_embeddings(self, skip_gram=False):
        """Return the nodes embeddings (word2vec)."""
        if not Graph.__w2v_model_ir_graph:
            self.__load_word2vec_model_ir_graph(skip_gram)

        unknown = Graph.__w2v_model_ir_graph.wv['unknown']

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            label = data["attr"].replace(' ', '')
            if label in Graph.__w2v_model_ir_graph.wv.vocab:
                nodes.append(Graph.__w2v_model_ir_graph[label])
            else:
                nodes.append(unknown)

        return np.array(nodes)

    def get_edge_str_list(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edges.append(
                (
                    nodes_keys.index(node1),
                    data["attr"],
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edges_and_types(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        types = []
        for node1, node2, data in self.G.edges(data=True):
            edges.append(
                (
                    nodes_keys.index(node1),
                    nodes_keys.index(node2)
                )
            )
            types.append(data["attr"])

        return edges, types

    def get_edges_dataFrame(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        source = []
        target = []
        type = []
        for node1, node2, data in self.G.edges(data=True):
            source.append(nodes_keys.index(node1))
            target.append(nodes_keys.index(node2))
            type.append(data["attr"])

        return pd.DataFrame({'source': source, 'target': target, 'type': type})

    def get_edge_list(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edges.append(
                (
                    nodes_keys.index(node1),
                    self.__edge_types.index(data["attr"]),
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edge_list_embeddings(self):
        """Return the edges and embeddings."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        emb = {
            0: np.array([0.0]),
            1: np.array([0.25]),
            2: np.array([0.5]),
            3: np.array([0.75]),
            4: np.array([1.0]),
            5: np.array([1.25]),
            6: np.array([1.50])
        }

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = self.__edge_types.index(data["attr"])
            edges.append(
                (
                    nodes_keys.index(node1),
                    emb[edge_type] if edge_type in emb else np.array([1.75]),
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edge_list_int2vec_embeddings(self):
        """Return the edges and inst2vec embeddings."""
        if not Graph.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        nodes_keys = list(self.__get_node_attr_dict().keys())

        augmented = {'!ECFG': 8569,
                     '!ECALL': 8570,
                     '!EDATA': 8571,
                     '!EMEM': 8572,
                     '!EBB': 8573,
                     '!EAST': 8574,
                     '!EIN': 8575,
                     '!EUNK': 8576}

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                (
                    nodes_keys.index(node1),
                    Graph.__inst2vec_embeddings[augmented[edge_type]]
                    if edge_type in augmented
                    else Graph.__inst2vec_embeddings[augmented['!EUNK']],
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edges_embeddings(self):
        """Return the edges embeedings."""
        emb = {
            0: np.array([0.0]),
            1: np.array([0.25]),
            2: np.array([0.5]),
            3: np.array([0.75]),
            4: np.array([1.0]),
            5: np.array([1.25]),
            6: np.array([1.50])
        }

        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = self.__edge_types.index(data["attr"])
            edges.append(
                emb[edge_type] if edge_type in emb else np.array([1.75])
            )

        return np.array(edges)

    def get_edges_inst2vec_embeddings(self):
        """Return the edges inst2vec embeddings."""
        if not Graph.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        augmented = {'!ECFG': 8569,
                     '!ECALL': 8570,
                     '!EDATA': 8571,
                     '!EMEM': 8572,
                     '!EBB': 8573,
                     '!EAST': 8574,
                     '!EIN': 8575,
                     '!EUNK': 8576}

        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                Graph.__inst2vec_embeddings[augmented[edge_type]]
                if edge_type in augmented
                else Graph.__inst2vec_embeddings[augmented['!EUNK']]
            )

        return np.array(edges)

    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2 in self.G.edges():
            edges.append(
                [
                    nodes_keys.index(node1),
                    nodes_keys.index(node2)
                ]
            )

        return csc_matrix(edges)

    def size(self):
        """Return the size of the graph."""
        return len(self.G)

    def draw(self, path=None, with_legend=False):
        """Create a dot graph."""
        # Copy graph object because attr modifications
        # for a cleaner view are needed.
        G = self.G

        # Add node labels.
        for (n, data) in G.nodes(data=True):
            G.nodes[n]["fillcolor"] = "gray"
            G.nodes[n]["style"] = "filled"

            if "attr" in data:
                if type(data["attr"]) is tuple:
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]

                if label == "function":
                    G.nodes[n]["label"] = "root"
                    G.nodes[n]["shape"] = "tab"
                    G.nodes[n]["fillcolor"] = "violet"
                else:
                    G.nodes[n]["label"] = label

            if "inst" in data or "insts" in data:
                G.nodes[n]["shape"] = "rectangle"
                G.nodes[n]["fillcolor"] = "cyan"

            if "value" in data:
                G.nodes[n]["shape"] = "diamond"

            if data["attr"] == "bb":
                G.nodes[n]["shape"] = "egg"
                G.nodes[n]["fillcolor"] = "orange3"

        # Add edge colors.
        edge_colors_by_types = {
            "ast": "black",
            "cfg": "blue",
            "data": "crimson",
            "mem": "pink",
            "call": "green",
            "in": "orange"
        }
        edge_colors_available = ["yellow"]

        for u, v, key, data in G.edges(keys=True, data=True):
            edge_type = data["attr"]
            if edge_type not in edge_colors_by_types:
                edge_colors_by_types[edge_type] = edge_colors_available.pop(0)

            G[u][v][key]["color"] = edge_colors_by_types[edge_type]

        # Create dot graph.
        graphviz_graph = nx.drawing.nx_agraph.to_agraph(G)

        # Add Legend.
        if with_legend:
            edge_types_used = set()
            for (u, v, key, data) in G.edges(keys=True, data=True):
                edge_type = data["attr"]
                edge_types_used.add(edge_type)

            subgraph = graphviz_graph.subgraph(name="cluster", label="Edges")
            for edge_type, color in edge_colors_by_types.items():
                if edge_type in edge_types_used:
                    subgraph.add_node(edge_type,
                                      color="invis",
                                      fontcolor=color)

        graphviz_graph.layout("dot")
        return graphviz_graph.draw(path)

    def networkx(self):
        """Return a Networkx representation."""
        # We create a new graph object because attr modifications.
        G = nx.MultiDiGraph()

        node_idx = 0
        nodes = {}

        # Add node labels.
        for (n, data) in self.G.nodes(data=True):
            if n not in nodes:
                nodes[n] = node_idx
                node_idx += 1
            if "attr" in data:
                if type(data["attr"]) is tuple:
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]
                G.add_node(nodes[n], attr=label)
            if "root" in data:
                G.add_node(nodes[n], root="root")
            if "inst" in data:
                if type(data["inst"]) is tuple:
                    inst = "\n".join(data["inst"])
                else:
                    inst = data["inst"]
                if G.has_node(nodes[n]):
                    G.nodes[nodes[n]]['instruction'] = inst
                else:
                    G.add_node(nodes[n], instruction=inst)
            if "value" in data:
                if type(data["value"]) is tuple:
                    value = "\n".join(data["value"])
                else:
                    value = data["value"]
                if G.has_node(nodes[n]):
                    G.nodes[nodes[n]]['value'] = value
                else:
                    G.add_node(nodes[n], value=value)

        # Add edge type.
        edge_types = {
            "ast": 0,
            "cfg": 1,
            "data": 2,
            "mem": 3,
            "call": 4,
            "in": 5
        }

        for u, v, _, data in self.G.edges(keys=True, data=True):
            G.add_edge(nodes[u], nodes[v], label=edge_types[data["attr"]]
                       if data["attr"] in edge_types else 6)

        return G

    def networkx_inst2vec(self):
        """Return a Networkx representation using embeddings (int2vec)."""
        if not Graph.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        # We create a new graph object because attr modifications.
        G = nx.MultiDiGraph()

        augmented = {'!UNK': Graph.__inst2vec_embeddings[8564],
                     '!IDENTIFIER': Graph.__inst2vec_embeddings[8565],
                     '!IMMEDIATE': Graph.__inst2vec_embeddings[8566],
                     '!MAGIC': Graph.__inst2vec_embeddings[8567],
                     '!BB': Graph.__inst2vec_embeddings[8568]}
        node_idx = 0
        nodes = {}

        # Add node labels.
        for (n, data) in self.G.nodes(data=True):
            if n not in nodes:
                nodes[n] = node_idx
                node_idx += 1
            if "value" in data:
                G.add_node(nodes[n], embeddings=augmented['!IMMEDIATE'])
            elif "inst" in data:
                if type(data["inst"]) is tuple:
                    inst = "\n".join(data["inst"])
                else:
                    inst = data["inst"]
                preprocessed, _ = i2v_pre.preprocess([[inst]])
                preprocessed = i2v_pre.PreprocessStatement(preprocessed[0][0])
                if preprocessed in Graph.__inst2vec_dictionary:
                    embeddings = Graph.__inst2vec_dictionary[preprocessed]
                    embeddings = Graph.__inst2vec_embeddings[embeddings]
                else:
                    embeddings = augmented['!UNK']
                G.add_node(nodes[n], embeddings=embeddings)
            elif "attr" in data:
                if type(data["attr"]) is tuple:
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]
                if label == 'bb':
                    G.add_node(nodes[n], embeddings=augmented['!BB'])
                elif label == 'function':
                    G.add_node(nodes[n], embeddings=augmented['!MAGIC'])
                else:
                    G.add_node(nodes[n], embeddings=augmented['!IDENTIFIER'])

        # Add edge colors.
        edge_types = {
            "ast": np.array([0.0]),  # 0
            "cfg": np.array([0.25]),  # 1
            "data": np.array([0.5]),  # 2
            "mem": np.array([0.75]),  # 3
            "call": np.array([1.0]),  # 4
            "in": np.array([1.25])  # 5
        }

        for u, v, _, data in self.G.edges(keys=True, data=True):
            G.add_edge(nodes[u], nodes[v], embeddings=edge_types[data["attr"]]
                       if data["attr"] in edge_types else np.array([1.50]))

        return G
