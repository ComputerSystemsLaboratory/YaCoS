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
from yacos.essentials import IO


class BestK:
    """Find the best k sequences.

    Fast and effective orchestration of compiler optimizations
     for automatic performance tuning
    Z. Pan and R. Eigenmann
    International Symposium on Code Generation and Optimization
    2006
    10.1109/CGO.2006.38
    """
    __version__ = '1.0.0'

    __flags = None

    # {key: 'seq': list}}
    __results = {}

    # {key: int}
    # The sequence 'key' is the better for 'int' benchmarks.
    __covering = {}

    @dataclass
    class Flags:
        """BestK flags.

        training_directory : str

        baseline_directory : str
        """

        training_directory: str
        baseline_directory: str

    def __init__(self,
                 training_directory,
                 baseline_directory):
        """Initialize a BestK object.

        Parameters
        ----------
        training_directory : str

        baseline_directory : str
        """
        self.__flags = self.Flags(training_directory,
                                  baseline_directory)

    def __get_maximum(self,
                      dictionary):
        """Get the maximum entry.

        Parameter
        ---------
        dictionary : dict
        """
        rank = [(len(dictionary[key]), key) for key in dictionary.keys()]
        rank.sort(reverse=True)

        maximum = rank[0][0]
        entries = [key for (len_, key) in rank if len_ == maximum]
        if len(entries) == 1:
            return entries[0]

        best_entry = []
        for entry in entries:
            improvements = [imp for _, imp in dictionary[entry]]
            best_entry.append((sum(improvements), entry))

        best_entry.sort(reverse=True)
        return best_entry[0][1]

    def __program_in_dictionary(self,
                                benchmark,
                                dictionary_list):
        """Verify whether program is in dictionary or not.

        Parameters
        ----------
        benchmark : str

        dictionary: list
            A dictionary entry
        """
        index = -1
        counter = 0
        for training, _ in dictionary_list:
            if training == benchmark:
                index = counter
                break
            counter += 1
        return index

    @property
    def results(self):
        """Getter."""
        return self.__results

    @property
    def covering(self):
        """Getter."""
        return self.__covering

    def run(self,
            training_benchmarks,
            compiler,
            baseline,
            k):
        """Best-k.

        Parameter
        --------
        training_benchmarks : list

        compiler : str

        baseline : str

        k : int
            Number of sequences
        """
        # Create the dictionary
        dictionary = {}
        best_sequences = {}
        for training_benchmark in training_benchmarks:
            index = training_benchmark.find('.')
            bench_dir = training_benchmark[:index]
            bench_name = training_benchmark[index+1:]

            training_dir = os.path.join(self.__flags.training_directory,
                                        bench_dir)
            baseline_dir = os.path.join(self.__flags.baseline_directory,
                                        bench_dir)

            training_sequences = IO.load_yaml(
                '{}/{}.yaml'.format(training_dir,
                                    bench_name)
            )

            if not training_sequences:
                continue

            # Baseline goal value
            b_goal_value = IO.load_yaml_or_fail(
                '{}/{}.yaml'.format(baseline_dir,
                                    bench_name)
            )
            b_goal_value = b_goal_value[compiler][baseline]['goal']

            # For each sequence
            for seq in training_sequences.keys():
                if seq not in dictionary.keys():
                    dictionary[seq] = []
                    best_sequences[seq] = training_sequences[seq]['seq']

                goal_value = training_sequences[seq]['goal']

                # Store the fitness
                if goal_value < b_goal_value:
                    diff = b_goal_value - goal_value
                    improvement = (diff / b_goal_value) * 100
                    dictionary[seq].append((training_benchmark, improvement))

        # Find the best dictionary entries
        if dictionary:
            bestk = []
            self.__covering = {}
            for _ in range(k):

                progs = []
                for _, data in dictionary.items():
                    progs += [p for p, _ in data if p not in progs]
                if len(progs) == 0:
                    break

                key = self.__get_maximum(dictionary)
                dictionary_entry = dictionary[key].copy()
                self.__covering[key] = len(dictionary_entry)

                bestk.append(key)

                for key, data in dictionary.items():
                    for program, improvement in dictionary_entry:
                        index = self.__program_in_dictionary(program,
                                                             data)
                        if index > -1:
                            del dictionary[key][index]

            # Store the best k sequences
            self.__results = {}
            for best in bestk:
                self.__results[best] = {'x': best_sequences[best]}
