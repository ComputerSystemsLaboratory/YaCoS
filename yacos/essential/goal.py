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
import sys

from statistics import mean
from absl import logging as lg

from yacos.essential import IO
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


class Goal:
    """Static class that manipulates goals."""

    __version__ = '1.0.0'

    @staticmethod
    def llvm_instructions(benchmark_directory):
        """Evaluate for number of LLVM instructions.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The number of LLVM instructions.
        """
        # Instantiate the LLVM driver.
        driver = LLVMDriver([])
        # Instantiate the builder.
        builder = R.LLVMInstsBuilder(driver)

        # Extract the number of instructions
        source = '{}/a.out_o.bc'.format(benchmark_directory)
        if not os.path.isfile(source):
            return float('inf')

        extractionInfo = builder.ir_to_info(source)
        goal_value = [data.instructions
                      for data in extractionInfo.functionInfos]
        return sum(goal_value)

    @staticmethod
    def binary_size(benchmark_directory):
        """Extract the binary size.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The binary size.
        """
        data = IO.load_yaml('{}/binary_size.yaml'.format(benchmark_directory))
        if data:
            return data
        else:
            return float('inf')

    @staticmethod
    def code_size(benchmark_directory):
        """Extract the code size.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The binary size.
        """
        data = IO.load_yaml('{}/code_size.yaml'.format(benchmark_directory))
        if data:
            goal_value = 0
            for _, value in data.items():
                goal_value += value
            return goal_value
        else:
            return float('inf')

    @staticmethod
    def text_size(benchmark_directory):
        """Extract the text size.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The binary size.
        """
        data = IO.load_yaml('{}/code_size.yaml'.format(benchmark_directory))
        if data:
            return data['text']
        else:
            return float('inf')

    @staticmethod
    def data_size(benchmark_directory):
        """Extract the data size.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The binary size.
        """
        data = IO.load_yaml('{}/code_size.yaml'.format(benchmark_directory))
        if data:
            return data['data']
        else:
            return float('inf')

    @staticmethod
    def bss_size(benchmark_directory):
        """Extract the bss size.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The binary size.
        """
        data = IO.load_yaml('{}/code_size.yaml'.format(benchmark_directory))
        if data:
            return data['bss']
        else:
            return float('inf')

    @staticmethod
    def compile_time(benchmark_directory):
        """Load the compile time.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The compile time.
        """
        data = IO.load_yaml('{}/compile_time.yaml'.format(benchmark_directory))
        if data:
            goal_value = mean([val for val in data])
        else:
            goal_value = float('inf')
        return goal_value

    @staticmethod
    def runtime(benchmark_directory):
        """Load the compile time.

        Parameters
        ---------
        benchmark_directory : str

        tool: str
            Execution tool

        Return
        ------
        value : float
            The compile time.
        """
        data = IO.load_yaml('{}/runtime.yaml'.format(benchmark_directory))
        if data:
            goal_value = mean([val for val in data])
        else:
            goal_value = float('inf')
        return goal_value

    @staticmethod
    def cycles(benchmark_directory):
        """Load the compile time.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The compile time.
        """
        data = IO.load_yaml('{}/cycles.yaml'.format(benchmark_directory))
        if data:
            goal_value = mean([val for val in data])
        else:
            goal_value = float('inf')
        return goal_value

    @staticmethod
    def instructions(benchmark_directory):
        """Load the compile time.

        Parameters
        ---------
        benchmark_directory : str

        Return
        ------
        value : float
            The compile time.
        """
        data = IO.load_yaml('{}/instructions.yaml'.format(benchmark_directory))
        if data:
            goal_value = mean([val for val in data])
        else:
            goal_value = float('inf')
        return goal_value

    @staticmethod
    def goal(goal):
        """Load the value of the goal."""
        if goal == 'llvm_instructions':
            return Goal.llvm_instructions
        elif goal == 'binary_size':
            return Goal.binary_size
        elif goal == 'code_size':
            return Goal.code_size
        elif goal == 'text_size':
            return Goal.text_size
        elif goal == 'data_size':
            return Goal.data_size
        elif goal == 'bss_size':
            return Goal.bss_size
        elif goal == 'compile_time':
            return Goal.compile_time
        elif goal == 'runtime':
            return Goal.runtime
        elif goal == 'cycles':
            return Goal.cycles
        elif goal == 'instructions':
            return Goal.instructions
        else:
            lg.error('Invalid goal {}'.format(goal))
            sys.exit(1)

    @staticmethod
    def prepare_goal(goals,
                     weights):
        """Create a dictionary.

        Parameters
        ----------
        goals : list

        weights : list

        Return
        ------
        goals : dict
            {goal: weight}
        """
        valid_goals = ['llvm_instructions',
                       'binary_size',
                       'code_size',
                       'text_size',
                       'data_size',
                       'bss_size',
                       'compile_time',
                       'runtime',
                       'cycles',
                       'instructions']

        for goal in goals:
            if goal not in valid_goals:
                lg.error('Invalid goal {}'.format(goal))
                sys.exit(1)

        if len(goals) != len(weights):
            lg.error('|goals| != |weights|')
            sys.exit(1)

        goals_ = {}
        value = 0.0
        for i, goal in enumerate(goals):
            weight = float(weights[i])
            goals_[goal] = weight
            value += weight

        if not (value > 0.1 and value <= 1.0):
            lg.error('Goal weight should be 1.0')
            sys.exit(1)

        return goals_

    @staticmethod
    def has_compile_time(goals):
        """Verify if compile_time is a goal."""
        return 'compile_time' in goals.keys()

    @staticmethod
    def only_compile_time_goal(goals):
        """Verify if the goals do not need to execute the benchmark."""
        inter = set(['cycles', 'instructions', 'runtime']).intersection(goals)
        return not bool(inter)

    @staticmethod
    def has_dynamic_goal(goals):
        """Verify if the YaCoS needs to execute the benchmark."""
        inter = set(['cycles', 'instructions', 'runtime']).intersection(goals)
        return bool(inter)
