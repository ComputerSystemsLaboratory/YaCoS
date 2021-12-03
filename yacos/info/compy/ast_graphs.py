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

import networkx as nx

from yacos.info.compy.extractors.extractors import Visitor
from yacos.info.compy.extractors.extractors import ClangDriver
from yacos.info.compy.extractors.extractors import ClangExtractor
from yacos.info.compy.extractors.extractors import clang
from yacos.info.compy import common


def find_node(G, attr):
    """Find a specific node."""
    for node in G.nodes(data=True):
        if node[1]['attr'] == attr:
            return node[0]
    return False


def filter_type(type):
    """Filter the type of the node."""
    if "[" in type or "]" in type:
        return "arrayType"
    elif "(" in type or ")" in type:
        return "fnType"
    elif "int" in type:
        return "intType"
    elif "float" in type:
        return "floatType"
    else:
        return "type"


class ASTVisitor(Visitor):
    """Abstract Syntax Tree Visitor."""

    def __init__(self):
        """Initialize an AST Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["ast"]
        self.node_types = ["fdecl", "arg", "stm", "root", "arel"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        """Visit method."""
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, type="fdecl", attr="FunctionDecl")
            for arg in v.args:
                self.G.add_node(arg,
                                type="arg",
                                attr=("ParmVarDecl", arg.type))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, type="stm", attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

            # Root node.
            node = find_node(self.G, "TranslationUnitDecl")
            if node:
                self.G.add_edge(node, v, attr="ast")
            else:
                self.G.add_node("TranslationUnitDecl",
                                attr="TranslationUnitDecl",
                                type="root",
                                root="root")
                self.G.add_edge("TranslationUnitDecl", v, attr="ast")

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, type="arel", attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")


class ASTDataVisitor(Visitor):
    """Abstract Syntax Tree + Data Visitor."""

    def __init__(self):
        """Initialize an ASTData Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["ast", "data"]
        self.node_types = ["fdecl", "arg", "stm", "root", "arel", "rrel"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        """Visit method."""
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, type="fdecl", attr="FunctionDecl")
            for arg in v.args:
                self.G.add_node(arg,
                                type="arg",
                                attr=("ParmVarDecl", arg.type))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, type="stm", attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

            # Root node.
            node = find_node(self.G, "TranslationUnitDecl")
            if node:
                self.G.add_edge(node, v, attr="ast")
            else:
                self.G.add_node("TranslationUnitDecl",
                                attr="TranslationUnitDecl",
                                type="root",
                                root="root")
                self.G.add_edge("TranslationUnitDecl", v, attr="ast")

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, type="arel", attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")
            for ref_rel in v.ref_relations:
                self.G.add_node(ref_rel,
                                type="rrel",
                                attr=(filter_type(ref_rel.type)))
                self.G.add_edge(v, ref_rel, attr="data")


class ASTDataCFGVisitor(Visitor):
    """Abstract Syntax Tree + Data + CFG Visitor."""

    def __init__(self):
        """Initialize an ASTDataCFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["ast", "cfg", "in", "data"]
        self.node_types = ["fdecl", "arg", "stm", "root", "arel", "rrel", "cfg"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        """Visit Method."""
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, type="fdecl", attr="FunctionDecl")
            for arg in v.args:
                self.G.add_node(arg,
                                type="arg",
                                attr=("ParmVarDecl", arg.type))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, type="stm", attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

            for cfg_b in v.cfgBlocks:
                self.G.add_node(cfg_b, type="cfg", attr="cfg")
                for succ in cfg_b.successors:
                    self.G.add_edge(cfg_b, succ, attr="cfg")
                    self.G.add_node(succ, type="cfg", attr="cfg")
                for stmt in cfg_b.statements:
                    self.G.add_edge(stmt, cfg_b, attr="in")
                    self.G.add_node(stmt, type="cfg", attr=(stmt.name))

            # Root node.
            node = find_node(self.G, "TranslationUnitDecl")
            if node:
                self.G.add_edge(node, v, attr="ast")
            else:
                self.G.add_node("TranslationUnitDecl",
                                attr="TranslationUnitDecl",
                                type="root",
                                root="root")
                self.G.add_edge("TranslationUnitDecl", v, attr="ast")

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, type="arel", attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")
            for ref_rel in v.ref_relations:
                self.G.add_node(ref_rel,
                                type="rrel",
                                attr=(filter_type(ref_rel.type)))
                self.G.add_edge(v, ref_rel, attr="data")


class ASTGraphBuilder(common.RepresentationBuilder):
    """AST Graph Builder."""

    def __init__(self, clang_driver=None):
        """Initialize a Graph Builder."""
        common.RepresentationBuilder.__init__(self)

        if clang_driver:
            self.__clang_driver = clang_driver
        else:
            self.__clang_driver = ClangDriver(
                ClangDriver.ProgrammingLanguage.C,
                ClangDriver.OptimizationLevel.O3,
                [],
                ["-Wall"],
            )
        self.__extractor = ClangExtractor(self.__clang_driver)

        self.__graphs = []

    def source_to_info(self, filename, additional_include_dir=None):
        """Extract information to build the representation."""
        if additional_include_dir:
            self.__clang_driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.GraphFromSource(filename)
        if additional_include_dir:
            self.__clang_driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def info_to_representation(self, info, visitor=ASTDataVisitor):
        """Build the representation for each function."""
        vis = visitor()
        info.accept(vis)

        for (n, data) in vis.G.nodes(data=True):
            attr = data["attr"]
            if attr not in self._tokens:
                self._tokens[attr] = 1
            self._tokens[attr] += 1

        return common.Graph(vis.G,
                            self.get_tokens(),
                            vis.node_types,
                            vis.edge_types)
