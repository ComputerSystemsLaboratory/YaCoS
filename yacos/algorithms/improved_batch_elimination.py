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


class ImprovedBatchElimination:
    """Improved Batch Elimination.

    Improved batch elimination: A fast algorithm to identify and
    remove harmful compiler optimizations
    E. Daniel de Lima and A. Faustino da Silva
    Latin American Computing Conference
    2015
    10.1109/CLEI.2015.7360010
    """

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """ImprovedBatchElimination flags.

        points : int
            The points to evaluate.

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

        points: int
        goals: dict
        compiler: str
        benchmarks_directory: str
        working_set: int
        times: int
        tool: str
        verify_output: bool

    def __init__(self,
                 points,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize an ImprovedBatchElimination object.

        points : int
            The points to evaluate.

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
        self.__flags = self.Flags(points,
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

    def run(self,
            benchmark,
            sequences):
        """Run Improved Batch elimination algorithm.

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

            best_sequence = sequence[:]
            best_goal_value = goal_value

            # Store the results
            self.__results[counter] = {'seq': best_sequence,
                                       'goal': best_goal_value}

            counter += 1

            prob = [0.0 for i in range(len(sequence))]
            factor = [0.0 for i in range(len(sequence))]

            # Evaluate N points to find the poor optimizations
            for S in range(len(sequence)):
                bool_vector = [1 for i in range(len(sequence))]
                bool_vector[S] = 0

                new_sequence = Sequence.remove_passes(sequence,
                                                      bool_vector)
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

                # If the fitness decreases, turn of the optimization
                if (goal_value < best_goal_value):
                    factor[S] = goal_value

            M = max(factor)

            if (M == 0.0):
                # Store the results
                self.__results[counter] = {'seq': best_sequence,
                                           'goal': best_goal_value}
            else:
                for S in range(len(sequence)):
                    if (factor[S] > 0.0):
                        prob[S] = factor[S] / M

                R = 0.0
                interval = 1.0 / (self.__flags.points - 1)
                for j in range(self.__flags.points):
                    b = [1 for i in range(len(sequence))]
                    for h in range(len(sequence)):
                        if (prob[h] >= R):
                            b[h] = 0

                    # Remove all poor optimizations and calculate the fitness
                    new_sequence = Sequence.remove_passes(sequence,
                                                          b)

                    # Evaluate the final sequence
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

                    if (goal_value < best_goal_value):
                        best_goal_value = goal_value
                        best_sequence = new_sequence[:]

                    R = R + interval

            # Store the results
            self.__results[counter] = {'seq': best_sequence,
                                       'goal': best_goal_value}

            counter += 1
