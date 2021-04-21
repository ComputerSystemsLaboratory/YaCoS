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


class IterativeElimination:
    """Iterative Elimination.

    Fast and effective orchestration of compiler optimizations
     for automatic performance tuning
    Z. Pan and R. Eigenmann
    International Symposium on Code Generation and Optimization
    2006
    10.1109/CGO.2006.38
    """

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """IterativeElimination flags.

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
        """Initialize a IterativeElimination object.

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

    def run(self,
            benchmark,
            sequences):
        """Run Iterative elimination algorithm.

        Argument
        ---------
        benchmark : str

        sequences : dict
        """
        self.__results = {}
        counter = 0

        # Package.Benchmark
        index = benchmark.find('.')

        # Benchmark directory
        bench_dir = os.path.join(self.__flags.benchmarks_directory,
                                 benchmark[:index],
                                 benchmark[index+1:])

        for _, seq_data in sequences.items():
            # Transform the sequence into a string
            sequence = seq_data['seq']

            # Package.Benchmark
            index = benchmark.find('.')

            # Evaluate the original sequence
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

            # Create a list of N elements
            final_bool_vector = [1 for _ in range(len(sequence))]
            search_space = [i for i in range(len(sequence))]

            # The algorithn
            while True:
                poors = []

                new_sequence = Sequence.remove_passes(sequence,
                                                      final_bool_vector)
                # Evaluate the original sequence
                baseline_value = Engine.evaluate(
                    self.__flags.goals,
                    Sequence.name_pass_to_string(new_sequence),
                    self.__flags.compiler,
                    bench_dir,
                    self.__flags.working_set,
                    self.__flags.times,
                    self.__flags.tool,
                    self.__flags.verify_output
                )

                for S in search_space:
                    if (not final_bool_vector[S]):
                        continue

                    local_vector = final_bool_vector[:]
                    local_vector[S] = 0

                    new_sequence = Sequence.remove_passes(sequence,
                                                          local_vector)

                    # Evaluate the original sequence
                    goal_value = Engine.evaluate(
                        self.__flags.goals,
                        Sequence.name_pass_to_string(new_sequence),
                        self.__flags.compiler,
                        bench_dir,
                        self.__flags.working_set,
                        self.__flags.times,
                        self.__flags.tool,
                        self.__flags.verify_output
                    )

                    if (goal_value < baseline_value):
                        poors.append((goal_value, S))

                if (poors):
                    poors.sort()
                    final_bool_vector[poors[0][1]] = 0
                    search_space.remove(poors[0][1])
                    del poors[0]

                if (poors == []):
                    break

            # Remove all poor optimizations and calculate the fitness
            final_sequence = Sequence.remove_passes(sequence,
                                                    final_bool_vector)

            # Evaluate the final sequence
            goal_value = Engine.evaluate(
                self.__flags.goals,
                Sequence.name_pass_to_string(final_sequence),
                self.__flags.compiler,
                bench_dir,
                self.__flags.working_set,
                self.__flags.times,
                self.__flags.tool,
                self.__flags.verify_output
            )

            # Store the results
            self.__results[counter] = {'seq': final_sequence,
                                       'goal': goal_value}

            counter += 1
