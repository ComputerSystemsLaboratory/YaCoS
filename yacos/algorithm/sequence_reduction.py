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


class SequenceReduction:
    """Create a small sequence."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    # {
    #   0: {'seq': list, 'goal': float} # original sequence
    #   1: {'seq': list, 'goal': float} # small sequence
    # }
    __results = {}

    @dataclass
    class Flags:
        """ReduceSequence flags.

        goals : dict
            {goal: weight}

        compiler : str
            The compiler to use.

        benchmarks_directory : str

        working_set : int

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
        """Initialize a SequenceReduction object.

        Parameters
        ----------
        goals : dict
            {goal: weight}

        compiler : str
            The compiler to use.

        benchmarks_directory : str

        working_set : int

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

    def run(self,
            sequence,
            benchmark):
        """Sequence Reduction algorithm.

        Suresh Purini and Lakshya Jain.
        Finding Good Optimization Sequences Covering Program Space.
        TACO.
        2013

        Parameter
        --------
        sequence : list

        benchmark : str
        """
        # Package.Benchmark
        index = benchmark.find('.')

        # Benchmark directory
        bench_dir = os.path.join(self.__flags.benchmarks_directory,
                                 benchmark[:index],
                                 benchmark[index+1:])

        # Calculate the initial value of the goal.
        goal_value = Engine.evaluate(self.__flags.goals,
                                     Sequence.name_pass_to_string(sequence),
                                     self.__flags.compiler,
                                     bench_dir,
                                     self.__flags.working_set,
                                     self.__flags.times,
                                     self.__flags.tool,
                                     self.__flags.verify_output)

        # Store the initial value of the goal.
        self.__results[0] = {'seq': sequence,
                             'goal': goal_value}

        # Sequence Reduction algorithm
        lst_best_sequence = sequence.copy()
        best_goal_value = goal_value
        change = True

        while change:

            change = False
            bestseqlen = len(lst_best_sequence)

            for i in range(bestseqlen):

                vector = [1 for i in range(bestseqlen)]
                vector[i] = 0

                lst_new_sequence = Sequence.remove_passes(
                    lst_best_sequence,
                    vector
                )

                goal_value = Engine.evaluate(
                    self.__flags.goals,
                    Sequence.name_pass_to_string(lst_new_sequence),
                    self.__flags.compiler,
                    bench_dir,
                    self.__flags.working_set,
                    self.__flags.times,
                    self.__flags.tool,
                    self.__flags.verify_output
                )

                if goal_value <= best_goal_value:
                    best_goal_value = goal_value
                    lst_best_sequence = lst_new_sequence[:]
                    change = True
                    break

        goal_value = Engine.evaluate(
            self.__flags.goals,
            Sequence.name_pass_to_string(lst_best_sequence),
            self.__flags.compiler,
            bench_dir,
            self.__flags.working_set,
            self.__flags.times,
            self.__flags.tool,
            self.__flags.verify_output
        )

        # Store the final value of the goal.
        self.__results[1] = {'seq': lst_best_sequence,
                             'goal': goal_value}
