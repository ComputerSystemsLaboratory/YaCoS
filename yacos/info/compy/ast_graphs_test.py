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

import networkx as nx

from absl import logging as lg

from yacos.info.compy.extractors.extractors import Visitor
from yacos.info.compy.extractors.extractors import clang
from yacos.info.compy.ast_graphs import ASTGraphBuilder
from yacos.info.compy.ast_graphs import ASTVisitor
from yacos.info.compy.ast_graphs import ASTDataVisitor
from yacos.info.compy.ast_graphs import ASTDataCFGVisitor


def verify_data_dir():
    top_dir = os.path.join(os.environ.get('HOME'), '.local')
    if not os.path.isdir(os.path.join(top_dir, 'yacos')):
        lg.error('YaCoS data does not exist.')
        sys.exit(1)


# Construction
def test_construct_with_custom_visitor():
    """Test custom vistor."""
    class CustomVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.edge_types = []
            self.G = nx.DiGraph()

        def visit(self, v):
            if not isinstance(v, clang.graph.ExtractionInfo):
                self.G.add_node(v, attr=type(v))

    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_1fn_2.c')

    builder = ASTGraphBuilder()
    info = builder.source_to_info(filename)
    ast = builder.info_to_representation(info, CustomVisitor)

    assert len(ast.G) == 14


# Attributes
def test_get_node_list():
    """Test node list."""

    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_1fn_2.c')

    builder = ASTGraphBuilder()
    info = builder.source_to_info(filename)
    ast = builder.info_to_representation(info, ASTDataVisitor)
    nodes = ast.get_node_list()

    assert len(nodes) == 15


def test_get_edge_list():
    """Test edge list."""

    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_1fn_2.c')

    builder = ASTGraphBuilder()
    info = builder.source_to_info(filename)
    ast = builder.info_to_representation(info, ASTDataVisitor)
    edges = ast.get_edge_list()

    assert len(edges) > 0

    assert type(edges[0][0]) is int
    assert type(edges[0][1]) is int
    assert type(edges[0][2]) is int


# Plot
def test_plot(tmpdir):
    """Test plot."""

    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_fib.c')

    for visitor in [ASTDataVisitor]:
        builder = ASTGraphBuilder()
        info = builder.source_to_info(filename)
        graph = builder.info_to_representation(info, ASTDataVisitor)

        outfile = os.path.join(tmpdir, str(visitor.__name__) + ".png")
        graph.draw(path=outfile, with_legend=True)

        assert os.path.isfile(outfile)

    # os.system('xdg-open ' + str(tmpdir))


# All visitors
def test_all_visitors():
    """All visitor."""

    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_1fn_2.c')

    for visitor in [ASTVisitor, ASTDataVisitor, ASTDataCFGVisitor]:
        builder = ASTGraphBuilder()
        info = builder.source_to_info(filename)
        ast = builder.info_to_representation(info, visitor)

        assert ast
