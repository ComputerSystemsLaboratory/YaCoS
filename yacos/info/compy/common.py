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
import random as rd
import collections

import networkx as nx
import pygraphviz as pgv

from absl import logging as lg
from gensim.models import Word2Vec, Doc2Vec
from scipy.sparse import csc_matrix
from yacos.essential import IO
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

    __w2v_dir = 'yacos/data/word2vec'
    __d2v_dir = 'yacos/data/doc2vec'

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

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_syntax_seq = Word2Vec.load(
                    os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_syntax_token_kind(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_syntax_token_kind_skip_gram.model'
        else:
            MODEL = 'w2v_syntax_token_kind_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_syntax_token_kind = Word2Vec.load(
                    os.path.join(top_dir,  self.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_syntax_token_kind_variable(self,
                                                         skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_syntax_token_kind_variable_skip_gram.model'
        else:
            MODEL = 'w2v_syntax_token_kind_variable_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_syntax_token_kind_variable = Word2Vec.load(
                    os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_llvm_seq(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_llvm_seq_skip_gram.model'
        else:
            MODEL = 'w2v_llvm_seq_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_llvm_seq = Word2Vec.load(
                    os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __load_doc2vec_model_syntax_seq(self):
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_seq.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__d2v_model_syntax_seq = Doc2Vec.load(
                    os.path.join(top_dir, self.__d2v_dir, MODEL)
        )

    def __load_doc2vec_model_syntax_token_kind(self):
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__d2v_model_syntax_token_kind = Doc2Vec.load(
                    os.path.join(top_dir, self.__d2v_dir, MODEL)
        )

    def __load_doc2vec_model_syntax_token_kind_variable(self):
        """Load a doc2vec model."""
        MODEL = 'd2v_syntax_token_kind_variable.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__d2v_model_syntax_token_kind_variable = Doc2Vec.load(
                    os.path.join(top_dir, self.__d2v_dir, MODEL)
        )

    def __load_doc2vec_model_llvm_seq(self):
        """Load a doc2vec model."""
        MODEL = 'd2v_llvm_seq.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__d2v_model_llvm_seq = Doc2Vec.load(
                    os.path.join(top_dir, self.__d2v_dir, MODEL)
        )

    def get_token_list(self):
        """Return the list of tokens."""
        node_ints = [self.__token_types.index(tok_str) for tok_str in self.S]
        return node_ints

    def get_token_name_list(self):
        """Return the list of tokens."""
        node_names = [tok_str for tok_str in self.S]
        return node_names

    def get_word2vec_list(self,
                          sequence_type='syntax_seq',
                          skip_gram=False):
        """Return the word2vec for each seq."""
        if sequence_type == 'syntax_seq':
            if not self.__w2v_model_syntax_seq:
                self.__load_word2vec_model_syntax_seq(skip_gram)
            model = self.__w2v_model_syntax_seq
        elif sequence_type == 'syntax_token_kind':
            if not self.__w2v_model_syntax_token_kind:
                self.__load_word2vec_model_syntax_token_kind(skip_gram)
            model = self.__w2v_model_syntax_token_kind
        elif sequence_type == 'syntax_token_kind_variable':
            if not self.__w2v_model_syntax_token_kind_variable:
                self.__load_word2vec_model_syntax_token_kind_variable(
                    skip_gram
                )
            model = self.__w2v_model_syntax_token_kind_variable
        elif sequence_type == 'llvm_seq':
            if not self.__w2v_model_llvm_seq:
                self.__load_word2vec_model_llvm_seq(skip_gram)
            model = self.__w2v_model_llvm_seq
        else:
            lg.error('Sequence type does not exist.')
            sys.error(1)

        unknown = model.wv['unknown']
        tokens = [tok_str for tok_str in self.S]
        nodes = []
        for token in tokens:
            if token.lower() in model.wv.vocab:
                nodes.append(model.wv[token.lower()])
            else:
                nodes.append(unknown)

        return np.array(nodes), np.array(unknown)

    def get_doc2vec(self, sequence_type='syntax_seq'):
        """Return the doc2vec features."""
        if sequence_type == 'syntax_seq':
            if not self.__d2v_model_syntax_seq:
                self.__load_doc2vec_model_syntax_seq()
            model = self.__d2v_model_syntax_seq
        elif sequence_type == 'syntax_token_kind':
            if not self.__d2v_model_syntax_token_kind:
                self.__load_doc2vec_model_syntax_token_kind()
            model = self.__d2v_model_syntax_token_kind
        elif sequence_type == 'syntax_token_kind_variable':
            if not self.__d2v_model_syntax_token_kind_variable:
                self.__load_doc2vec_model_syntax_token_variable_kind()
            model = self.__d2v_model_syntax_token_variable_kind
        elif sequence_type == 'llvm_seq':
            if not self.__d2v_model_llvm_seq:
                self.__load_doc2vec_model_llvm_seq()
            model = self.__d2v_model_llvm_seq
        else:
            lg.error('Sequence type does not exist.')
            sys.error(1)

        tokens = [tok_str.lower() for tok_str in self.S]
        return model.infer_vector(tokens)

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
    __w2v_model_asm_graph = None
    __inst2vec_dictionary = None
    __inst2vec_embeddings = None
    __ir2vec_dictionary = None
    __ir2vec_embeddings = None

    __boo_ast = None
    __boo_ir = None
    __boo_asm = None

    __w2v_dir = 'yacos/data/word2vec'
    __i2v_dir = 'yacos/data/inst2vec'
    __ir2v_dir = 'yacos/data/ir2vec'
    __boo_dir = 'yacos/data/bag_of_words'

    def __init__(self, graph, node_attrs, node_types, edge_types):
        """Initialize a Graph representation."""
        self.G = graph
        self.__node_attr = node_attrs
        self.__node_types = node_types
        self.__edge_types = edge_types

    def __load_inst2vec_data(self):
        """Load the dictionary and embeddings."""
        DICTIONARY = 'inst2vec_augmented_dictionary.pickle'
        EMBEDDINGS = 'inst2vec_augmented_embeddings.pickle'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        filename = os.path.join(top_dir, self.__i2v_dir, DICTIONARY)
        self.__inst2vec_dictionary = IO.load_pickle_or_fail(filename)

        filename = os.path.join(top_dir, self.__i2v_dir, EMBEDDINGS)
        self.__inst2vec_embeddings = IO.load_pickle_or_fail(filename)

    def __load_ir2vec_data(self):
        """Load the dictionary and embeddings."""
        DICTIONARY = 'ir2vec_augmented_dictionary.pickle'
        EMBEDDINGS = 'ir2vec_augmented_embeddings.pickle'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        filename = os.path.join(top_dir, self.__ir2v_dir, DICTIONARY)
        self.__ir2vec_dictionary = IO.load_pickle_or_fail(filename)

        filename = os.path.join(top_dir, self.__ir2v_dir, EMBEDDINGS)
        self.__ir2vec_embeddings = IO.load_pickle_or_fail(filename)

    def __load_word2vec_model_ast(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_ast_skip_gram.model'
        else:
            MODEL = 'w2v_ast_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_ast = Word2Vec.load(os.path.join(top_dir,
                                                          self.__w2v_dir,
                                                          MODEL))

    def __load_word2vec_model_ir_graph(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_ir_graph_skip_gram.model'
        else:
            MODEL = 'w2v_ir_graph_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_ir_graph = Word2Vec.load(
            os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __load_word2vec_model_asm_graph(self, skip_gram=False):
        """Load a word2vec model."""
        if skip_gram:
            MODEL = 'w2v_asm_graph_skip_gram.model'
        else:
            MODEL = 'w2v_asm_graph_cbow.model'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        self.__w2v_model_asm_graph = Word2Vec.load(
            os.path.join(top_dir, self.__w2v_dir, MODEL)
        )

    def __load_boo_ast(self):
        """Load bag of words (AST) dictionary."""
        DICTIONARY = 'ast_bag_of_words.pickle'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        filename = os.path.join(top_dir, self.__boo_dir, DICTIONARY)
        self.__boo_ast = IO.load_pickle_or_fail(filename)

    def __load_boo_ir(self):
        """Load bag of words (LLVM  IR) dictionary."""
        DICTIONARY = 'ir_bag_of_words.pickle'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        filename = os.path.join(top_dir, self.__boo_dir, DICTIONARY)
        self.__boo_ir = IO.load_pickle_or_fail(filename)

    def __load_boo_asm(self):
        """Load bag of words (Assembly) dictionary."""
        DICTIONARY = 'asm_bag_of_words.pickle'

        top_dir = os.path.join(os.environ.get('HOME'), '.local')
        if not os.path.isdir(os.path.join(top_dir, 'yacos')):
            lg.error('YaCoS data does not exist.')
            sys.exit(1)

        filename = os.path.join(top_dir, self.__boo_dir, DICTIONARY)
        self.__boo_asm = IO.load_pickle_or_fail(filename)

    def __get_node_attr_dict(self):
        """Return the node attributes."""
        return collections.OrderedDict(self.G.nodes(data="attr",
                                                    default="N/A"))

    def __opcode_embeddings(self, opcode):
        """Return the opcode embeddings."""
        rd.seed(opcode)
        return rd.random()

    def get_node_str_list(self):
        """Return the node attributes."""
        node_strs = list(self.__get_node_attr_dict().values())
        return node_strs

    def get_node_list(self):
        """Return the nodes."""
        node_strs = list(self.__get_node_attr_dict().values())
        node_ints = [self.__node_attrs.index(n_str) for n_str in node_strs]
        return node_ints

    def get_nodes_inst2vec_embeddings(self, type_name=False):
        """Return the nodes embeddings (int2vec)."""
        nodes_keys = list(self.__get_node_attr_dict().keys())
        if not self.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            type = data["type"]
            if type == "root":
                embeddings = self.__inst2vec_embeddings[
                                self.__inst2vec_dictionary['!MAGIC']
                             ]
            elif type == "cdfgplusbb":
                embeddings = self.__inst2vec_embeddings[
                                self.__inst2vec_dictionary['!BB']
                             ]
            elif type == "insn" or type == "bb":
                emb = []
                for inst in data['insts']:
                    if isinstance(inst, tuple):
                        inst_ = "\n".join(inst)
                    else:
                        inst_ = inst
                    try:
                        preprocessed, _ = i2v_pre.preprocess([[inst_]])
                        preprocessed = i2v_pre.PreprocessStatement(
                                        preprocessed[0][0]
                                        )
                    except Exception:
                        #
                        # TODO: preprocess do not handle:
                        #
                        # call void asm sideeffect "movgs $0,ibar0", "r,~{dirflag},~{fpsr},~{flags}"(i64 %167) #2, !srcloc !2
                        #
                        preprocessed = '!UNK'
                    if preprocessed in self.__inst2vec_dictionary:
                        embeddings = self.__inst2vec_embeddings[
                                    self.__inst2vec_dictionary[preprocessed]
                                     ]
                    else:
                        index = self.__inst2vec_dictionary['!UNK']
                        embeddings = self.__inst2vec_embeddings[index]
                    emb.append(embeddings)
                embeddings = [sum(x) for x in zip(*emb)]
            elif type == "id" or type == "data":
                embeddings = self.__inst2vec_embeddings[
                                self.__inst2vec_dictionary['!IDENTIFIER']
                             ]
            elif type == "imm":
                embeddings = self.__inst2vec_embeddings[
                                    self.__inst2vec_dictionary['!IMMEDIATE']
                             ]
            else:
                lg.error("Node type ({}) does not exist.".format(type))
                sys.exit(1)

            nodes.append((nodes_keys.index(n),
                          type if type_name else self.__node_types.index(type),
                          np.array(embeddings)))

        return nodes

    def get_nodes_ir2vec_embeddings(self, type_name=False):
        """Return the nodes embeddings (ir2vec)."""
        nodes_keys = list(self.__get_node_attr_dict().keys())
        if not self.__ir2vec_dictionary:
            self.__load_ir2vec_data()

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            type = data["type"]
            if type == "root":
                embeddings = self.__ir2vec_embeddings[
                                self.__ir2vec_dictionary['!MAGIC']
                             ]
            elif type == "cdfgplusbb":
                embeddings = self.__ir2vec_embeddings[
                                self.__ir2vec_dictionary['!BB']
                             ]
            elif type == "insn" or type == "bb":
                emb = []
                for inst in data['insts']:
                    if isinstance(inst, tuple):
                        inst_ = "\n".join(inst)
                    else:
                        inst_ = inst
                    try:
                        preprocessed, _ = i2v_pre.preprocess([[inst_]])
                        preprocessed = i2v_pre.PreprocessStatement(
                                        preprocessed[0][0]
                                        )
                    except Exception:
                        #
                        # TODO: preprocess do not handle:
                        #
                        # call void asm sideeffect "movgs $0,ibar0", "r,~{dirflag},~{fpsr},~{flags}"(i64 %167) #2, !srcloc !2
                        #
                        preprocessed = '!UNK'
                    if preprocessed in self.__ir2vec_dictionary:
                        embeddings = self.__ir2vec_embeddings[
                                    self.__ir2vec_dictionary[preprocessed]
                                     ]
                    else:
                        index = self.__ir2vec_dictionary['!UNK']
                        embeddings = self.__ir2vec_embeddings[index]
                    emb.append(embeddings)
                embeddings = [sum(x) for x in zip(*emb)]
            elif type == "id" or type == "data":
                embeddings = self.__ir2vec_embeddings[
                                self.__ir2vec_dictionary['!IDENTIFIER']
                             ]
            elif type == "imm":
                embeddings = self.__ir2vec_embeddings[
                                    self.__ir2vec_dictionary['!IMMEDIATE']
                             ]
            else:
                lg.error("Node type ({}) does not exist.".format(type))
                sys.exit(1)

            nodes.append((nodes_keys.index(n),
                          type if type_name else self.__node_types.index(type),
                          np.array(embeddings)))

        return nodes

    def get_nodes_bag_of_words_embeddings(self,
                                          graph_type='ir',
                                          compact=False,
                                          type_name=False):
        """Return the nodes embeddings (bag of words)."""
        if graph_type == 'ast':
            if not self.__boo_ast:
                self.__load_boo_ast()
            boo = self.__boo_ast
        elif graph_type == 'asm':
            if not self.__boo_asm:
                self.__load_boo_asm()
            boo = self.__boo_asm
        elif graph_type == 'ir':
            if not self.__boo_ir:
                self.__load_boo_ir()
            boo = self.__boo_ir
        else:
            lg.error('Boo type does not exist.')
            sys.exit(1)

        """
        bag_of_words.pickle

        classes:
          terminator: 0
          unary: 1
          binary: 2
          ...
        instructions:
          ret:
            class: 0
            pos: 0
          ...
          fneg:
            class: 1
            pos: 11
          ...
        """
        nodes_keys = list(self.__get_node_attr_dict().keys())

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            if compact:
                embeddings = [0 for _ in range(len(boo['classes']))]
            else:
                embeddings = [0 for _ in range(len(boo['instructions']))]

            type = data["type"]
            if type == "root":
                if compact:
                    embeddings[boo['instructions']['magic']['class']] += 1
                else:
                    embeddings[boo['instructions']['magic']['pos']] += 1
            elif type == "cdfgplusbb":
                if compact:
                    embeddings[boo['instructions']['bb']['class']] += 1
                else:
                    embeddings[boo['instructions']['bb']['pos']] += 1
            elif type == "insn" or type == "bb":
                if compact:
                    for opcode in data['opcodes']:
                        embeddings[boo['instructions'][opcode]['class']] += 1
                else:
                    for opcode in data['opcodes']:
                        embeddings[boo['instructions'][opcode]['pos']] += 1
            elif type == "id" or type == "data":
                if compact:
                    embeddings[
                                boo['instructions']['identifier']['class']
                              ] += 1
                else:
                    embeddings[
                                boo['instructions']['identifier']['pos']
                              ] += 1
            elif type == "imm":
                if compact:
                    embeddings[boo['instructions']['immediate']['class']] += 1
                else:
                    embeddings[boo['instructions']['immediate']['pos']] += 1
            else:
                lg.error("Node type ({}) does not exist.".format(type))
                sys.exit(1)

            nodes.append((nodes_keys.index(n),
                          type if type_name else self.__node_types.index(type),
                          np.array(embeddings)))

        return nodes

    def get_nodes_word2vec_embeddings(self,
                                      graph_type='ir',
                                      skip_gram=False,
                                      type_name=False):
        """Return the nodes embeddings (word2vec)."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        if graph_type == 'asm':
            if not self.__w2v_model_asm_graph:
                self.__load_word2vec_model_asm_graph(skip_gram)
            model = self.__w2v_model_asm_graph
        elif graph_type == 'ast':
            if not self.__w2v_model_ast_graph:
                self.__load_word2vec_model_ast_graph(skip_gram)
            model = self.__w2v_model_ast_graph
        elif graph_type == 'ir':
            if not self.__w2v_model_ir_graph:
                self.__load_word2vec_model_ir_graph(skip_gram)
            model = self.__w2v_model_ir_graph
        else:
            lg.error('Graph_type does not exist.')
            sys.exit(1)

        unknown = model.wv['unknown']

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            type = data["type"]
            if type == "root":
                embeddings = model.wv['magic']
            elif type == "cdfgplusbb":
                embeddings = model.wv['bb']
            elif type == "insn" or type == "bb":
                emb = []
                for opcode in data['opcodes']:
                    if isinstance(opcode, tuple):
                        label = "\n".join(opcode)
                    else:
                        label = opcode
                    if label.lower() in model.wv.vocab:
                        emb.append(model.wv[label.lower()])
                    else:
                        emb.append(unknown)
                embeddings = [sum(x) for x in zip(*emb)]
            elif type == "id" or type == "data":
                embeddings = model.wv['identifier']
            elif type == "imm":
                embeddings = model.wv['immediate']
            else:
                lg.error("Node type ({}) does not exist.".format(type))
                sys.exit(1)

            nodes.append((nodes_keys.index(n),
                          type if type_name else self.__node_types.index(type),
                          np.array(embeddings)))

        return nodes

    def get_nodes_opcode_embeddings(self, type_name=False):
        """Return the nodes embeddings (word2vec)."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        nodes = []
        for (n, data) in self.G.nodes(data=True):
            type = data["type"]
            if type == "root":
                embeddings = self.__opcode_embeddings('magic')
            elif type == "cdfgplusbb":
                embeddings = self.__opcode_embeddings('bb')
            elif type == "insn" or type == "bb":
                embeddings = 0
                for opcode in data['opcodes']:
                    embeddings += self.__opcode_embeddings(opcode)
            elif type == "id" or type == "data":
                embeddings = self.__opcode_embeddings('identifier')
            elif type == "imm":
                embeddings = self.__opcode_embeddings('immediate')
            elif type == "fdecl":
                embeddings = self.__opcode_embeddings('fdecl')
            elif type == "arg":
                embeddings = self.__opcode_embeddings(data["attr"][1])
            elif type == "stm":
                embeddings = self.__opcode_embeddings(data["attr"])
            elif type == "arel":
                embeddings = self.__opcode_embeddings(data["attr"])
            elif type == "rrel":
                embeddings = self.__opcode_embeddings(data["attr"])
            elif type == "cfg":
                embeddings = self.__opcode_embeddings(data["attr"])
            else:
                lg.error("Node type ({}) does not exist.".format(type))
                sys.exit(1)

            nodes.append((nodes_keys.index(n),
                          type if type_name else self.__node_types.index(type),
                          np.array(embeddings)))

        return nodes

    def get_edges_dataFrame(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        source = []
        target = []
        type_ = []
        for node1, node2, data in self.G.edges(data=True):
            source.append(nodes_keys.index(node1))
            target.append(nodes_keys.index(node2))
            type_.append(self.__edge_types.index(data["attr"]))

        return pd.DataFrame({'source': source,
                             'target': target,
                             'type': type_})

    def get_edges_str_dataFrame(self):
        """Return the edges."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        source = []
        target = []
        type_ = []
        for node1, node2, data in self.G.edges(data=True):
            source.append(nodes_keys.index(node1))
            target.append(nodes_keys.index(node2))
            type_.append(data["attr"])

        return pd.DataFrame({'source': source,
                             'target': target,
                             'type': type_})

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
            types.append(self.__edge_types.index(data["attr"]))

        return edges, types

    def get_edges_and_types_str(self):
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

    def get_edge_list_type_embeddings(self):
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

    def get_edge_list_inst2vec_embeddings(self):
        """Return the edges and inst2vec embeddings."""
        if not self.__inst2vec_dictionary:
            self.__load_inst2vec_data()

        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                (
                    nodes_keys.index(node1),
                    self.__inst2vec_embeddings[
                            self.__inst2vec_dictionary[edge_type]
                    ]
                    if edge_type in self.__inst2vec_dictionary
                    else self.__inst2vec_embeddings[
                                self.__inst2vec_dictionary['!EUNK']
                         ],
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edge_list_ir2vec_embeddings(self):
        """Return the edges and ir2vec embeddings."""
        if not self.__ir2vec_dictionary:
            self.__load_ir2vec_data()

        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                (
                    nodes_keys.index(node1),
                    self.__ir2vec_embeddings[
                        self.__ir2vec_dictionary[edge_type]
                    ]
                    if edge_type in self.__ir2vec_dictionary
                    else self.__ir2vec_embeddings[
                            self.__ir2vec_dictionary['!EUNK']
                         ],
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edge_list_word2vec_embeddings(self,
                                          graph_type='ir',
                                          skip_gram=False):
        """Return the nodes embeddings (word2vec)."""
        if graph_type == 'asm':
            if not self.__w2v_model_asm_graph:
                self.__load_word2vec_model_asm_graph(skip_gram)
            model = self.__w2v_model_asm_graph
        elif graph_type == 'ast':
            if not self.__w2v_model_ast_graph:
                self.__load_word2vec_model_ast_graph(skip_gram)
            model = self.__w2v_model_ast_graph
        elif graph_type == 'ir':
            if not self.__w2v_model_ir_graph:
                self.__load_word2vec_model_ir_graph(skip_gram)
            model = self.__w2v_model_ir_graph
        else:
            lg.error('Graph_type does not exist.')
            sys.exit(1)

        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = 'e{}'.format(data["attr"].lower())
            edges.append(
                (
                    nodes_keys.index(node1),
                    model.wv[edge_type]
                    if edge_type in model.wv.vocab
                    else model.wv['eunk'],
                    nodes_keys.index(node2)
                )
            )

        return edges

    def get_edge_list_bag_of_words_embeddings(self):
        """Return the nodes embeddings (bag of words)."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        edges = []
        for node1, node2, data in self.G.edges(data=True):
            edge_type = self.__edge_types.index(data["attr"])
            emb = [0 for _ in range(0, 7)]
            emb[edge_type] = 1

            edges.append(
                (
                    nodes_keys.index(node1),
                    emb,
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

        return edges

    def get_edges_inst2vec_embeddings(self):
        """Return the edges and inst2vec embeddings."""
        if not self.__inst2vec_dictionary:
            self.__load_ir2vec_data()

        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                self.__inst2vec_embeddings[
                    self.__inst2vec_dictionary[edge_type]
                ]
                if edge_type in self.__inst2vec_dictionary
                else self.__inst2vec_embeddings[
                        self.__inst2vec_dictionary['!EUNK']
                     ]
            )

        return edges

    def get_edges_ir2vec_embeddings(self):
        """Return the edges and ir2vec embeddings."""
        if not self.__ir2vec_dictionary:
            self.__load_ir2vec_data()

        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = '!E{}'.format(data["attr"].upper())
            edges.append(
                self.__ir2vec_embeddings[
                    self.__ir2vec_dictionary[edge_type]
                ]
                if edge_type in self.__ir2vec_dictionary
                else self.__ir2vec_embeddings[
                        self.__ir2vec_dictionary['!EUNK']
                     ]
            )

        return edges

    def get_edges_word2vec_embeddings(self,
                                      graph_type='ir',
                                      skip_gram=False):
        """Return the nodes embeddings (word2vec)."""
        if graph_type == 'asm':
            if not self.__w2v_model_asm_graph:
                self.__load_word2vec_model_asm_graph(skip_gram)
            model = self.__w2v_model_asm_graph
        elif graph_type == 'ast':
            if not self.__w2v_model_ast_graph:
                self.__load_word2vec_model_ast_graph(skip_gram)
            model = self.__w2v_model_ast_graph
        elif graph_type == 'ir':
            if not self.__w2v_model_ir_graph:
                self.__load_word2vec_model_ir_graph(skip_gram)
            model = self.__w2v_model_ir_graph
        else:
            lg.error('Graph_type does not exist.')
            sys.exit(1)

        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = 'e{}'.format(data["attr"].lower())
            edges.append(
                    model.wv[edge_type]
                    if edge_type in model.wv.vocab
                    else model.wv['eunk']
            )

        return edges

    def get_edges_bag_of_words_embeddings(self):
        """Return the nodes embeddings (bag of words)."""
        edges = []
        for _, _, data in self.G.edges(data=True):
            edge_type = self.__edge_types.index(data["attr"])
            emb = [0 for _ in range(0, 7)]
            emb[edge_type] = 1

            edges.append(emb)

        return edges

    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        nodes_keys = list(self.__get_node_attr_dict().keys())

        row = []
        col = []
        data = []
        for node1, node2 in self.G.edges():
            row.append(nodes_keys.index(node1))
            col.append(nodes_keys.index(node2))
            data.append(1)

        return coo_matrix((np.array(data), (np.array(row), np.array(col))),
                          shape=(len(self.G), len(self.G)))

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
                if isinstance(data["attr"], tuple):
                    label = "\n".join(data["attr"])
                else:
                    label = data["attr"]

                if label == "function":
                    G.nodes[n]["label"] = "root"
                    G.nodes[n]["shape"] = "tab"
                    G.nodes[n]["fillcolor"] = "violet"
                else:
                    G.nodes[n]["label"] = label

            if "insts" in data:
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
