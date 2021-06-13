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

from dataclasses import dataclass

from yacos.essential import Sequence
from yacos.essential import Engine


class GraphFromSequences:
    """Graph From Sequences.

    A Graph-based Model for Building Optimization Sequences
    A Study Case on Code Size Reduction

    Nilton and Anderson
    SBLP - 2021
    """

    __version__ = '2.1.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """BatchElimination flags.

        sequences : dict

        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times : int
            Execution times

        tool : str
            Execution tool

        verify_output : bool
            The goal is valid only if the execution status is OK.
        """

        goals: dict
        compiler: str
        benchmarks_directory: str
        working_set: int
        times: int
        tool: str
        verify_output: bool

    def __init__(self,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize a BatchElimination object.

        goals : dict
            {goal: weight}

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times : int
            Execution times

        tool : str
            Execution tool

        verify_output : bool
            The goal is valid only if the execution status is OK.
        """
        self.__flags = self.Flags(goals,
                                  compiler,
                                  benchmarks_directory,
                                  working_set,
                                  times,
                                  tool,
                                  verify_output)

    @property
    def results(self):
        """Getter."""
        return self.__results

    def build(self):
        """Build the graph (model)."""
        pass

    def traversal_similar(self):
        """
        This traversal algorithm selects the next vertex based on
        the similarity between the training and testprograms. The next
        vertex, V.succ, is that labeled withthe optimization from 𝑆𝑖,
        which belongs to the mostsimilar training program 𝑃𝑖.
        """
        pass

    def traversal_cost(self):
        """
        This traversal algorithm selects the next vertex following the lowest
        cost edge.
        """
        pass

    def traversal_weighted(self):
        """
        This traversal algorithm is based on the twoprevious algorithms.
        In this traversal, both strategies are applied (similar and cost) and
        the next vertex is chosen by the strategythat provides the lowest value
        between similarity andedge cost.
        """
        pass

    def traversal_backtracking(self):
        """
        This traversal algorithm evaluates the performance of the sub-sequences
        during the traversal, i.e. when the algorithm selects a next vertex it
        evaluates the new sub-sequence. This task aims to guide the traversal.
        """
        pass
