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

from yacos.info.compy.extractors.extractors import Visitor
from yacos.info.compy.extractors.extractors import ClangDriver
from yacos.info.compy.extractors.extractors import LLVMDriver
from yacos.info.compy.extractors.extractors import LLVMIRExtractor
from yacos.info.compy.extractors.extractors import llvm
from yacos.info.compy import common


def merge_after_element_on_condition(elements, element_conditions):
    """Merge elements.

    Ex.: If merged on conditions ['a'], ['a', 'b', 'c', 'a', 'e']
         becomes ['ab', 'c', 'ae']
    """
    for i in range(len(elements) - 2, -1, -1):
        if elements[i] in element_conditions:
            elements[i] = elements[i] + elements.pop(i + 1)

    return elements


def filer_elements(elements, element_filter):
    """Filter elements.

    Ex.: If filtered on elements [' '], ['a', ' ', 'c']
         becomes ['a', 'c']
    """
    return [element for element in elements if element not in element_filter]


def strip_elements(elements, element_filters):
    """Strip elements.

    Ex.: If stripped on elments [' '], ['a', ' b', 'c']
         becomes ['a', 'b', 'c']
    """
    ret = []
    for element in elements:
        for element_filter in element_filters:
            element = element.strip(element_filter)
        ret.append(element)

    return ret


def strip_function_name(elements):
    """Strip function name."""
    for i in range(len(elements) - 1):
        if elements[i] == "@":
            elements[i + 1] = "fn_0"

    return elements


def transform_elements(elements):
    """Transform elements."""
    elements = merge_after_element_on_condition(elements, ["%", "i"])
    elements = strip_elements(elements, ["\n", " "])
    elements = filer_elements(elements, ["", " ", "local_unnamed_addr"])
    return elements


class LLVMSeqVisitor(Visitor):
    """Sequence Visitor."""

    def __init__(self):
        """Init a Seq Visitor."""
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        """Visit method."""
        if isinstance(v, llvm.seq.FunctionInfo):
            self.S += strip_function_name(transform_elements(v.signature))

        if isinstance(v, llvm.seq.BasicBlockInfo):
            self.S += [v.name + ":"]

        if isinstance(v, llvm.seq.InstructionInfo):
            self.S += transform_elements(v.tokens)


class LLVMSeqBuilder(common.RepresentationBuilder):
    """Seq Builder."""

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
        info = self.__extractor.SeqFromSource(filename)
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
        info = self.__extractor.SeqFromIR(filename)

        return info

    def info_to_representation(self, info, visitor=LLVMSeqVisitor):
        """Build the representation for each function."""
        vis = visitor()
        info.accept(vis)

        for token in vis.S:
            if token not in self._tokens:
                self._tokens[token] = 1
            self._tokens[token] += 1

        return common.Sequence(vis.S, self.get_tokens())
