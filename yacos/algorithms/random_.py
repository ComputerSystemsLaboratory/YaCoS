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


class Random:
    """Random generates and evaluates random sequences."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """Random flags.

        nof_sequences : int
            The number of sequences.

        mininum : int
            The minimum length of the sequence.

        maximum : int
            The maximum length of the sequence.

        factor : int
            The times to appy to nof_sequences. (nof_sequences *= factor)

        ssa : bool
            Enable ssa?

        shuffle : bool
            Enable shuffle?

        update : bool
            Enable update?

        repetition : bool
            Enable repetition?

        original : bool
            Insert the orginal?

        passes_filename : str
            The yaml filename which describes the available passes.

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

        nof_sequences: int
        minimum: int
        maximum: int
        factor: int
        ssa: bool
        shuffle: bool
        update: bool
        repetition: bool
        original: bool
        passes_filename: str
        goals: dict
        compiler: str
        benchmarks_directory: str
        working_set: int
        times: int
        tool: str
        verify_output: bool

    def __init__(self,
                 nof_sequences,
                 minimum,
                 maximum,
                 factor,
                 ssa,
                 shuffle,
                 update,
                 repetition,
                 original,
                 passes_filename,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize a Random object.

        Parameters
        ----------
        nof_sequences : int
            The number of sequences.

        mininum : int
            The minimum length of the sequence.

        maximum : int
            The maximum length of the sequence.

        factor : int
            The times to appy to nof_sequences. (nof_sequences *= factor)

        ssa : bool
            Enable ssa?

        shuffle : bool
            Enable shuffle?

        update : bool
            Enable update?

        repetition : bool
            Enable repetition?

        original : bool
            Insert the orginal?

        passes_filename : str
            The yaml filename which describes the available passes.

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
        self.__flags = self.Flags(nof_sequences,
                                  minimum,
                                  maximum,
                                  factor,
                                  ssa,
                                  shuffle,
                                  update,
                                  repetition,
                                  original,
                                  passes_filename,
                                  goals,
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

    def run(self, benchmark):
        """Random algorithm.

        Parameter
        ---------
        benchmark: str
        """
        self.__results = {}
        counter = 0
        stop = False

        # Paackage.benchmark
        index = benchmark.find('.')
        # Benchmark directory
        bench_dir = os.path.join(self.__flags.benchmarks_directory,
                                 benchmark[:index],
                                 benchmark[index+1:])

        while True:
            # Create N sequences
            sequences = Sequence.create_random_sequences(
                self.__flags.nof_sequences,
                self.__flags.minimum,
                self.__flags.maximum,
                self.__flags.factor,
                self.__flags.ssa,
                self.__flags.shuffle,
                self.__flags.update,
                self.__flags.repetition,
                self.__flags.original,
                self.__flags.passes_filename
            )

            # For each sequence
            for _, data in sequences.items():
                sequence = data['seq']
                # Calculate the fitness
                if Sequence.exist(sequence, self.__results):
                    continue

                index = benchmark.find('.')

                goal_value = Engine.evaluate(
                    self.__flags.goals,
                    Sequence.name_pass_to_string(sequence),
                    self.__flags.compiler,
                    bench_dir,
                    self.__flags.working_set,
                    self.__flags.times,
                    self.__flags.tool,
                    self.__flags.verify_output
                )

                if goal_value == float('inf'):
                    continue

                # Store the results
                self.__results[counter] = {'seq': sequence,
                                           'goal': goal_value}

                counter += 1
                if counter == self.__flags.nof_sequences:
                    stop = True
                    break

            if stop:
                break
