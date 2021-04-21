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

from yacos.essentials import Sequence
from yacos.essentials import Engine


class CBR_Function:
    """CBR Function optimizes each function using a specific sequence."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #       'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """CBR flags.

        Parameters
        ----------
        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times: int
            Execution times

        tool : str
            Execution tool

        verify_output: bool
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
        """Initialize the arguments.

        Parameters
        ----------
        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times: int
            Execution times

        tool: str
            Execution tool

        verify_output: bool
            The goal is valid only if the execution status is OK.
        """
        # When the goal is obtained during compile time
        # and the working set is not defined during compilation,
        # we do not need the working set.
        self.__flags = self.Flags(goals,
                                  compiler,
                                  benchmarks_directory,
                                  working_set,
                                  times,
                                  tool,
                                  verify_output)

    def __extract_sequences(self,
                            training_sequences,
                            similarity,
                            nof_sequences=0):
        """Extract N sequences from training programs.

        Argument
        ---------
        training_sequences : dict
            {program: {'seq': X, 'goal': Y}}

        similarity : dict
            {function: {program: value, ..., program: value}}

        nof_sequences: int

        Return
        ------
        sequences : list
        """
        sequences = {}
        for function, similarity_ in similarity.items():
            rank_training_benchmarks = [
                (value, name)
                for name, value in similarity_.items()
            ]

            rank_training_benchmarks.sort()

            # Select N sequences from the most similar program
            benchmark = rank_training_benchmarks[0][1]
            sequences_ = training_sequences[benchmark]

            seqs_list = [
                seq_data['seq']
                for index, (_, seq_data) in enumerate(sequences_.items())
                if index < nof_sequences
            ]

            for counter in range(len(seqs_list)):
                if counter not in sequences:
                    sequences[counter] = {}
                sequences[counter][function] = Sequence.name_pass_to_string(
                        seqs_list[counter]
                )

        return sequences

    @property
    def results(self):
        """Getter."""
        return self.__results

    def run(self,
            benchmark,
            training_sequences,
            similarity,
            nof_sequences=0):
        """CBR algorithm.

        Argument
        ---------
        benchmark : str

        training_sequences : dict

        similarity : dict
            {program: value, ..., program: value}

        nof_sequences: int
        """
        # Package.Benchmark
        index = benchmark.find('.')

        # Benchmark directory
        bench_dir = os.path.join(self.__flags.benchmarks_directory,
                                 benchmark[:index],
                                 benchmark[index+1:])

        # Extract N sequences
        sequences = self.__extract_sequences(
            training_sequences,
            similarity,
            nof_sequences
        )

        self.__results = {}
        counter = 0

        # Evaluate the sequences
        for _, seq_data in sequences.items():
            # Extract functions
            Engine.extract_globals_and_functions(bench_dir)

            # Optimize each function and link
            Engine.compile_functions(bench_dir, seq_data)

            # Evaluate the sequence
            goal_value = Engine.only_evaluate(
                self.__flags.goals,
                bench_dir,
                self.__flags.working_set,
                self.__flags.times,
                self.__flags.tool,
                self.__flags.verify_output
            )

            if goal_value == float('inf'):
                continue

            # Store the results
            self.__results[counter] = {'seq': seq_data,
                                       'goal': goal_value}

            counter += 1
