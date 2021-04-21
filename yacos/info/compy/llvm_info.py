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

import sys
from absl import logging as lg

from yacos.info.compy.extractors.extractors import ClangDriver
from yacos.info.compy.extractors.extractors import LLVMDriver
from yacos.info.compy.extractors.extractors import LLVMIRExtractor


class LLVMNamesBuilder():
    """Globals and Functions names."""

    def __init__(self, driver=None):
        """Initialize the representation."""
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
        """Extract info from the source code."""
        if not isinstance(self.__driver, ClangDriver):
            lg.error('source_to_info needs ClangDriver')
            sys.exit(1)
        if additional_include_dir:
            self.__driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.NamesFromSource(filename)
        if additional_include_dir:
            self.__driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def ir_to_info(self, filename):
        """Extract info from the IR."""
        if not isinstance(self.__driver, LLVMDriver):
            lg.error('ir_to_info needs LLVMDriver')
            sys.exit(1)
        info = self.__extractor.NamesFromIR(filename)

        return info


class LLVMInstsBuilder():
    """LLVM number of instructions."""

    def __init__(self, driver=None):
        """Initialize the representation."""
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
        """Extract info from the source code."""
        if not isinstance(self.__driver, ClangDriver):
            lg.error('source_to_info needs ClangDriver')
            sys.exit(1)
        if additional_include_dir:
            self.__driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.InstsFromSource(filename)
        if additional_include_dir:
            self.__driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def ir_to_info(self, filename):
        """Extract info from the IR."""
        if not isinstance(self.__driver, LLVMDriver):
            lg.error('ir_to_info needs LLVMDriver')
            sys.exit(1)

        info = self.__extractor.InstsFromIR(filename)

        return info


class LLVMWLCostBuilder():
    """LLVM cost of functions."""

    def __init__(self, driver=None):
        """Initialize the representation."""
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
        """Extract info from the source code."""
        if not isinstance(self.__driver, ClangDriver):
            lg.error('source_to_info needs ClangDriver')
            sys.exit(1)
        if additional_include_dir:
            self.__driver.addIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )
        info = self.__extractor.WLCostFromSource(filename)
        if additional_include_dir:
            self.__driver.removeIncludeDir(
                additional_include_dir, ClangDriver.IncludeDirType.User
            )

        return info

    def ir_to_info(self, filename):
        """Extract info from the IR."""
        if not isinstance(self.__driver, LLVMDriver):
            lg.error('ir_to_info needs LLVMDriver')
            sys.exit(1)
        info = self.__extractor.WLCostFromIR(filename)

        return info
