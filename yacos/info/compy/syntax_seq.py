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

from yacos.info.compy.extractors.extractors import Visitor
from yacos.info.compy.extractors.extractors import ClangDriver
from yacos.info.compy.extractors.extractors import ClangExtractor
from yacos.info.compy.extractors.extractors import clang
from yacos.info.compy import common


class SyntaxSeqVisitor(Visitor):
    """Syntax Sequence Visitor."""

    def __init__(self):
        """Init a SyntaxSeq Visitor."""
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        """Visit method."""
        if isinstance(v, clang.seq.TokenInfo):
            self.S.append(v.name)


class SyntaxTokenkindVisitor(Visitor):
    """Syntax Token Visitor."""

    def __init__(self):
        """Init a Token Visitor."""
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        """Visit method."""
        if isinstance(v, clang.seq.TokenInfo):
            self.S.append(v.kind)


class SyntaxTokenkindVariableVisitor(Visitor):
    """Syntax Token + Variable Visitor."""

    def __init__(self):
        """Init a Token+Variable Visitor."""
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        """Visit method."""
        if isinstance(v, clang.seq.TokenInfo):
            if v.kind == "raw_identifier" and "var" in v.name:
                self.S.append(v.name)
            elif (
                v.name in ["for", "while", "do", "if", "else", "return"]
                or v.name in ["fn_0"]
                or v.name.startswith("int")
                or v.name.startswith("float")
            ):
                self.S.append(v.name)
            else:
                self.S.append(v.kind)


class SyntaxSeqBuilder(common.RepresentationBuilder):
    """Syntax Sequence builder."""

    def __init__(self, clang_driver=None):
        """Init the builder."""
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

    def source_to_info(self, filename, additional_include_dir=None):
        """Extract information to build the representation."""
        if additional_include_dir:
            self.__clang_driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.SeqFromSource(filename)
        if additional_include_dir:
            self.__clang_driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def info_to_representation(self, info, visitor=SyntaxSeqVisitor):
        """Build the representation for each function."""
        vis = visitor()
        info.accept(vis)

        for token in vis.S:
            if token not in self._tokens:
                self._tokens[token] = 1
            self._tokens[token] += 1

        return common.Sequence(vis.S, self.get_tokens())
