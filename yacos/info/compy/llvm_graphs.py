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

import sys

from absl import logging as lg
import networkx as nx

from yacos.info.compy.extractors.extractors import Visitor
from yacos.info.compy.extractors.extractors import ClangDriver
from yacos.info.compy.extractors.extractors import LLVMDriver
from yacos.info.compy.extractors.extractors import LLVMIRExtractor
from yacos.info.compy.extractors.extractors import llvm
from yacos.info.compy import common


def find_node(G, attr):
    """Find a node based on the attribute attr."""
    for node in G.nodes(data=True):
        if node[1]['attr'] == attr:
            return node[0]
    return False


def is_one_edge(visitor):
    """Verify the visitor."""
    return (isinstance(visitor, LLVMCFGCallCompactSingleEdgeVisitor)
            or isinstance(visitor, LLVMCFGCallCompactSingleEdgeNoRootVisitor)
            or isinstance(visitor, LLVMCDFGCompactSingleEdgeVisitor)
            or isinstance(visitor, LLVMCDFGCallCompactSingleEdgeVisitor)
            or isinstance(visitor, LLVMCDFGCallCompactSingleEdgeNoRootVisitor))


def has_edge(G, edge1, edge2, attr):
    """Verify if a edge exists."""
    try:
        edges = G.edges(edge1, data=True)
        for e1, e2, att in edges:
            if e2 == edge2 and att['attr'] == attr:
                return True
        return False
    except Exception:
        return False


class LLVMCFGVisitor(Visitor):
    """Control Flow Graph Visitor.
       1 instruction per basic block."""

    def __init__(self):
        """Initialize a CFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg"]
        self.node_types = ["insn"]
        self.G = nx.MultiDiGraph()
        self.root = False

    def visit(self, v):
        """Visit method."""
        # if isinstance(v, llvm.graph.FunctionInfo):
        #    # Function arg nodes.
        #    for arg in v.args:
        #        self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            if not self.root:
                self.G.add_node(v,
                                attr=(v.opcode),
                                opcodes=[(v.opcode)],
                                insts=[(v.instStr)],
                                type="insn",
                                root="root")
                self.root = True
            else:
                self.G.add_node(v,
                                attr=(v.opcode),
                                opcodes=[(v.opcode)],
                                insts=[(v.instStr)],
                                type="insn")


class LLVMCFGCompactVisitor(Visitor):
    """Control Flow Graph Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg"]
        self.node_types = ["bb"]
        self.G = nx.MultiDiGraph()
        self.root = False

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            if not self.root:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcodes=opcodes,
                                insts=insts,
                                type="bb",
                                root="root")
                self.root = True
            else:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcodes=opcodes,
                                type="bb",
                                insts=insts)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")


class LLVMCFGCallVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["insn", "root"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node,
                                v.entryInstruction,
                                attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v, node, attr="call")
                else:
                    self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget


class LLVMCFGCallNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["insn"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            # Call edges.
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget


class LLVMCFGCallCompactMultipleEdgesVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["bb", "root"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node,
                                v.entryInstruction.basicBlock,
                                attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v.basicBlock, node, attr="call")
                else:
                    self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        attr="call"
                    )
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock,
                                        v.basicBlock,
                                        attr="call")
                else:
                    self.calls_bb[v] = v.callTarget


class LLVMCFGCallCompactSingleEdgeVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block.
       Add only one edge."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["bb", "root"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node,
                                v.entryInstruction.basicBlock,
                                attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v.basicBlock, node, attr="call")
                else:
                    self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(
                        self.G,
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        "call"
                    ):
                        self.G.add_edge(
                            v.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            attr="call"
                        )
                    for exit in called_function.exitInstructions:
                        if not has_edge(
                            self.G,
                            exit.basicBlock,
                            v.basicBlock,
                            "call"
                        ):
                            self.G.add_edge(exit.basicBlock,
                                            v.basicBlock,
                                            attr="call")
                else:
                    self.calls_bb[v] = v.callTarget


class LLVMCFGCallCompactMultipleEdgesNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["bb"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        attr="call"
                    )
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock,
                                        v.basicBlock,
                                        attr="call")
                else:
                    self.calls_bb[v] = v.callTarget


class LLVMCFGCallCompactSingleEdgeNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.node_types = ["bb"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            # for arg in v.args:
            #    self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(
                        self.G,
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        "call"
                    ):
                        self.G.add_edge(
                            v.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            attr="call"
                        )
                    for exit in called_function.exitInstructions:
                        if not has_edge(
                            self.G,
                            exit.basicBlock,
                            v.basicBlock,
                            "call"
                        ):
                            self.G.add_edge(exit.basicBlock,
                                            v.basicBlock,
                                            attr="call")
                else:
                    self.calls_bb[v] = v.callTarget


class LLVMCDFGVisitor(Visitor):
    """Control-data Flow Graph Visitor."""

    def __init__(self):
        """Initialize a CDFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.node_types = ["insn", "data"]
        self.G = nx.MultiDiGraph()
        self.root = False

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            if not self.root:
                self.G.add_node(v,
                                attr=(v.opcode),
                                insts=[(v.instStr)],
                                type="insn",
                                root="root")
                self.root = True
            else:
                self.G.add_node(v,
                                attr=(v.opcode),
                                opcodes=[(v.opcode)],
                                insts=[(v.instStr)],
                                type="insn")

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGCompactMultipleEdgesVisitor(Visitor):
    """Control-data Flow Graph Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CDFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.node_types = ["bb", "data"]
        self.G = nx.MultiDiGraph()
        self.root = False

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst.basicBlock,
                                            memacc.inst.basicBlock,
                                            attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            if not self.root:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcodes=opcodes,
                                insts=insts,
                                type="bb",
                                root="root")
                self.root = True
            else:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcode=opcodes,
                                type="bb",
                                insts=insts)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_edge(operand.basicBlock,
                                    v.basicBlock,
                                    attr="data")


class LLVMCDFGCompactSingleEdgeVisitor(Visitor):
    """Control-data Flow Graph Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CDFG Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.node_types = ["bb", "data"]
        self.G = nx.MultiDiGraph()
        self.root = False

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            if not has_edge(
                                self.G,
                                dep.inst.basicBlock,
                                memacc.inst.basicBlock,
                                "mem"
                            ):
                                self.G.add_edge(dep.inst.basicBlock,
                                                memacc.inst.basicBlock,
                                                attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            if not self.root:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcodes=opcodes,
                                insts=insts,
                                type="bb",
                                root="root")
                self.root = True
            else:
                self.G.add_node(v,
                                attr=(v.fullName),
                                opcodes=opcodes,
                                insts=insts,
                                type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    if not has_edge(self.G, operand, v.basicBlock, "data"):
                        self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    if not has_edge(self.G,
                                    operand.basicBlock,
                                    v.basicBlock,
                                    "data"):
                        self.G.add_edge(operand.basicBlock,
                                        v.basicBlock,
                                        attr="data")


class LLVMCDFGCallVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["insn", "root", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node, v.entryInstruction, attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v, node, attr="call")
                else:
                    self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGCallNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["insn", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGCallCompactMultipleEdgesVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["bb", "root", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node,
                                v.entryInstruction.basicBlock,
                                attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst.basicBlock,
                                            memacc.inst.basicBlock,
                                            attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v.basicBlock, node, attr="call")
                else:
                    self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(
                                v.basicBlock,
                                called_function.entryInstruction.basicBlock,
                                attr="call"
                    )
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock,
                                        v.basicBlock,
                                        attr="call")
                else:
                    self.calls_bb[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    self.G.add_edge(operand,
                                    v.basicBlock,
                                    attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_edge(operand.basicBlock,
                                    v.basicBlock,
                                    attr="data")


class LLVMCDFGCallCompactSingleEdgeVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["bb", "root", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node,
                                v.entryInstruction.basicBlock,
                                attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            if not has_edge(self.G,
                                            dep.inst.basicBlock,
                                            memacc.inst.basicBlock,
                                            "mem"):
                                self.G.add_edge(dep.inst.basicBlock,
                                                memacc.inst.basicBlock,
                                                attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v.basicBlock, node, attr="call")
                else:
                    self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(
                        self.G,
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        "call"
                    ):
                        self.G.add_edge(
                            v.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            attr="call"
                        )
                    for exit in called_function.exitInstructions:
                        if not has_edge(
                            self.G,
                            exit.basicBlock,
                            v.basicBlock,
                            "call"
                        ):
                            self.G.add_edge(exit.basicBlock,
                                            v.basicBlock,
                                            attr="call")
                else:
                    self.calls_bb[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    if not has_edge(
                        self.G,
                        operand,
                        v.basicBlock,
                        "data"
                    ):
                        self.G.add_edge(operand,
                                        v.basicBlock,
                                        attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    if not has_edge(
                        self.G,
                        operand.basicBlock,
                        v.basicBlock,
                        "data"
                    ):
                        self.G.add_edge(operand.basicBlock,
                                        v.basicBlock,
                                        attr="data")


class LLVMCDFGCallCompactMultipleEdgesNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["bb", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst.basicBlock,
                                            memacc.inst.basicBlock,
                                            attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(
                                v.basicBlock,
                                called_function.entryInstruction.basicBlock,
                                attr="call"
                    )
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock,
                                        v.basicBlock,
                                        attr="call")
                else:
                    self.calls_bb[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    self.G.add_edge(operand,
                                    v.basicBlock,
                                    attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_edge(operand.basicBlock,
                                    v.basicBlock,
                                    attr="data")


class LLVMCDFGCallCompactSingleEdgeNoRootVisitor(Visitor):
    """Control-data Flow Graph + Call Visitor.
       Several instructions per basic block.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGCall Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.node_types = ["bb", "data"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls_bb = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            if not has_edge(
                                self.G,
                                dep.inst.basicBlock,
                                memacc.inst.basicBlock,
                                "mem"
                            ):
                                self.G.add_edge(dep.inst.basicBlock,
                                                memacc.inst.basicBlock,
                                                attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            insts = [instr.instStr for instr in v.instructions]
            opcodes = [instr.opcode for instr in v.instructions]

            self.G.add_node(v,
                            attr=(v.fullName),
                            opcodes=opcodes,
                            insts=insts,
                            type="bb")

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v,
                                succ.instructions[0].basicBlock,
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(
                        self.G,
                        v.basicBlock,
                        called_function.entryInstruction.basicBlock,
                        attr="call"
                    ):
                        self.G.add_edge(
                            v.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            attr="call"
                        )
                    for exit in called_function.exitInstructions:
                        if not has_edge(
                            self.G,
                            exit.basicBlock,
                            v.basicBlock,
                            attr="call"
                        ):
                            self.G.add_edge(exit.basicBlock,
                                            v.basicBlock,
                                            attr="call")
                else:
                    self.calls_bb[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    if not has_edge(
                        self.G,
                        operand,
                        v.basicBlock,
                        "data"
                    ):
                        self.G.add_edge(operand,
                                        v.basicBlock,
                                        attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    if not has_edge(
                        self.G,
                        operand.basicBlock,
                        v.basicBlock,
                        "data"
                    ):
                        self.G.add_edge(operand.basicBlock,
                                        v.basicBlock,
                                        attr="data")


class LLVMCDFGPlusVisitor(Visitor):
    """Control-data Flow Graph Plus Visitor."""

    def __init__(self):
        """Initialize a CDFGPlus Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call", "bb"]
        self.node_types = ["insn", "root", "data", "cdfgplusbb"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v
            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node, v.entryInstruction, attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # BB nodes
            self.G.add_node(v, attr="bb", type="cdfgplusbb")
            for instr in v.instructions:
                self.G.add_edge(instr, v, attr="bb")
            for succ in v.successors:
                self.G.add_edge(v, succ, attr="bb")

            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v, node, attr="call")
                else:
                    self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGPlusNoRootVisitor(Visitor):
    """Control-data Flow Graph Plus Visitor.
       This version has no root node."""

    def __init__(self):
        """Initialize a CDFGPlus Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call", "bb"]
        self.node_types = ["insn", "data", "cdfgplusbb"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="data")

            # Memory accesses
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # BB nodes
            self.G.add_node(v, attr="bb", type="cdfgplusbb")
            for instr in v.instructions:
                self.G.add_edge(instr, v, attr="bb")
            for succ in v.successors:
                self.G.add_edge(v, succ, attr="bb")

            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMProGraMLVisitor(Visitor):
    """ProGraML Visitor."""

    def __init__(self):
        """Initialize a ProGraML Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "call"]
        self.node_types = ["insn", "root", "imm", "id"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            node = find_node(self.G, "function")
            if node:
                self.G.add_edge(node, v.entryInstruction, attr="call")
            else:
                self.G.add_node(v, attr="function", type="root", root="root")
                self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="id")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            # Call edges.
            if v.opcode == "ret":
                node = find_node(self.G, "function")
                if node:
                    self.G.add_edge(v, node, attr="call")
                else:
                    self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.ConstantInfo
                ):
                    if isinstance(operand, llvm.graph.ConstantInfo) and operand.value:
                        self.G.add_node(operand,
                                        attr=(operand.type),
                                        value=(operand.value),
                                        type="imm")
                    else:
                        self.G.add_node(operand,
                                        attr=(operand.type),
                                        type="id")
                    self.G.add_edge(operand, v, attr="data")
                elif isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_node((v, operand),
                                    attr=(operand.type),
                                    type="id")
                    self.G.add_edge(operand, (v, operand), attr="data")
                    self.G.add_edge((v, operand), v, attr="data")


class LLVMProGraMLNoRootVisitor(Visitor):
    """ProGraML Visitor."""

    def __init__(self):
        """Initialize a ProGraML Visitor."""
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "call"]
        self.node_types = ["insn", "imm", "id"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type), type="id")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1],
                                succ.instructions[0],
                                attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v,
                            attr=(v.opcode),
                            opcodes=[(v.opcode)],
                            insts=[(v.instStr)],
                            type="insn")

            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v,
                                    called_function.entryInstruction,
                                    attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.ConstantInfo
                ):
                    if isinstance(operand, llvm.graph.ConstantInfo) and operand.value:
                        self.G.add_node(operand,
                                        attr=(operand.type),
                                        value=(operand.value),
                                        type="imm")
                    else:
                        self.G.add_node(operand,
                                        attr=(operand.type),
                                        type="id")
                    self.G.add_edge(operand, v, attr="data")
                elif isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_node((v, operand),
                                    attr=(operand.type),
                                    type="id")
                    self.G.add_edge(operand, (v, operand), attr="data")
                    self.G.add_edge((v, operand), v, attr="data")


class LLVMGraphBuilder(common.RepresentationBuilder):
    """Graph Builder."""

    def __init__(self, driver=None):
        """Init the builder."""
        common.RepresentationBuilder.__init__(self)

        if driver:
            self.__driver = driver
        else:
            self.__driver = ClangDriver(
                ClangDriver.ProgrammingLanguage.C,
                ClangDriver.OptimizationLevel.O3,
                [],
                ["-Wall"],
            )
        self.__extractor = LLVMIRExtractor(self.__driver)

    def source_to_info(self, filename, additional_include_dir=None):
        """Extract information to build the representation."""
        if not isinstance(self.__driver, ClangDriver):
            lg.error('source_to_info needs ClangDriver')
            sys.exit(1)
        if additional_include_dir:
            self.__driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.GraphFromSource(filename)
        if additional_include_dir:
            self.__driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def ir_to_info(self, filename):
        """Extract information to build the representation."""
        if not isinstance(self.__driver, LLVMDriver):
            lg.error('ir_to_info needs LLVMDriver')
            sys.exit(1)
        info = self.__extractor.GraphFromIR(filename)

        return info

    def info_to_representation(self, info, visitor=LLVMProGraMLVisitor):
        """Build the representation for each function."""
        vis = visitor()
        info.accept(vis)

        if 'calls' in vis.__dict__:
            for inst, call in vis.calls.items():
                called_function = (
                    vis.functions[call]
                    if call in vis.functions
                    else None
                )
                if called_function:
                    vis.G.add_edge(inst,
                                   called_function.entryInstruction,
                                   attr="call")
                    for exit in called_function.exitInstructions:
                        vis.G.add_edge(exit, inst, attr="call")
        elif 'calls_bb' in vis.__dict__:
            for inst, call in vis.calls_bb.items():
                called_function = (
                    vis.functions[call]
                    if call in vis.functions
                    else None
                )
                if called_function:
                    if is_one_edge(visitor):
                        if not has_edge(
                            vis.G,
                            inst.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            "call"
                        ):
                            vis.G.add_edge(
                                inst.basicBlock,
                                called_function.entryInstruction.basicBlock,
                                attr="call"
                            )
                    else:
                        vis.G.add_edge(
                            inst.basicBlock,
                            called_function.entryInstruction.basicBlock,
                            attr="call"
                        )
                    for exit in called_function.exitInstructions:
                        if is_one_edge(visitor):
                            if not has_edge(
                                vis.G,
                                exit.basicBlock,
                                inst.basicBlock,
                                "call"
                            ):
                                vis.G.add_edge(
                                    exit.basicBlock,
                                    inst.basicBlock,
                                    attr="call"
                                )
                        else:
                            vis.G.add_edge(exit.basicBlock,
                                           inst.basicBlock,
                                           attr="call")

        for (n, data) in vis.G.nodes(data=True):
            # print(n, data, flush=True)
            attr = data["attr"]
            if attr not in self._tokens:
                self._tokens[attr] = 1
            self._tokens[attr] += 1

        return common.Graph(vis.G,
                            self.get_tokens(),
                            vis.node_types,
                            vis.edge_types)
