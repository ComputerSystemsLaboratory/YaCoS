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
import shutil
import subprocess
from absl import logging as lg

from yacos.essentials import IO
from yacos.essentials import Goals
from yacos.info import compy as R
from yacos.info.compy.extractors import LLVMDriver


class classproperty(property):
    """class property decorator."""

    def __get__(self, cls, owner):
        """Decorate."""
        return classmethod(self.fget).__get__(None, owner)()


class Engine:
    """Static class to compile and run benchmarks."""

    __version__ = '1.0.0'

    __file_name_length = 25

    @classproperty
    def file_name_length(cls):
        """Getter."""
        return cls.__file_name_length

    @staticmethod
    def copy_tree(src, dst):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                Engine.copy_tree(s, d)
            else:
                if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                    shutil.copy2(s, d)

    @staticmethod
    def compile_(benchmark_directory,
                 compiler,
                 sequence,
                 working_set=0,
                 times=1):
        """Compile a benchmark.

        YaCoS use the tool make to compile a benchmark.
        So, the user should provide a Makefile.

        Parameters
        ----------
        benchmark_directory : str

        sequence : str
            The compiler optimizations to enable.

        compiler : str
            The compiler: clang, llvm, icc, gcc

        times: int

        working_set: int
            Some benchmarks define the working set during
            compilation. Examples: PolyBench and NPB.

        Return
        ------
        res : bool
            True if success, otherwise False.
        """
        # Prepare the command
        #
        # Some benchmarks (PolyBench, NPB, ...) define the working_set
        # during compilation. In such cases, compile.sh have
        # to handle this situation.
        cmdline = 'curr_dir=$PWD ; ' \
                  + 'cd {0} ; ' \
                  + './compile.sh {1} "{2}" {3} {4} ; ' \
                  + 'cd $curr_dir'
        cmdline = cmdline.format(benchmark_directory,
                                 compiler,
                                 sequence,
                                 working_set,
                                 times)

        # Invoke the compiler
        res = True
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            res = False
            lg.warning('Compile: {}'.format(benchmark_directory))

        # Return
        return res

    @staticmethod
    def compile_functions(benchmark_directory,
                          sequences,
                          sequence='',
                          working_set=0,
                          times=1):
        """Compile each function using a specific sequence.

        YaCoS use the tool make to compile a benchmark.
        So, the user should provide a Makefile.

        Parameters
        ----------
        benchmark_directory : str

        sequence : dict
            {function: sequence}
            The compiler optimizations to enable.

        sequence : str
            A sequence to apply after linking.

        times: int

        working_set: int
            Some benchmarks define the working set during
            compilation. Examples: PolyBench and NPB.

        Return
        ------
        res : bool
            True if success, otherwise False.
        """
        # This methods have to be invoked after extract_globals_and_functions.
        # In other words, it will not "extract" functions.

        # Step 1: Optimize each function
        Engine.optimize_functions(benchmark_directory,
                                  sequences)

        # Step 2: Generate the target code
        cmdline = 'curr_dir=$PWD ; '\
                  + 'cd {0} ; '\
                  + './compile.sh {1} "{2}" {3} {4} ; '\
                  + 'cd $curr_dir'
        cmdline = cmdline.format(benchmark_directory,
                                 'merge',
                                 sequence,
                                 working_set,
                                 times)

        # Invoke the compiler
        res = True
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            res = False
            lg.warning('Compile: {}'.format(benchmark_directory))

        # Return
        return res

    @staticmethod
    def execute(benchmark_directory,
                working_set,
                times,
                tool,
                verify_output,
                warmup_cache,
                runtime):
        """Execute the benchmark.

        Parameter
        ---------
        benchmark_directory : str

        working_set : int

        times : int

        tool: int
            The user can use the flag to invoke different
            types of execution: hyperfine, pin, perf, ...

        verify_output: bool
            The goal value is valid if and only if
            the output (execution) is correct.

        warmup_cache: bool

        runtime: int
            The runtime execution (timeout)

        Return
        ------
        value : float
            The number of instructions.
        """
        cmdline = 'curr_dir=$PWD ; '\
                  + 'cd {0} ; '\
                  + './execute.sh {1} {2} {3} {4} {5} {6} ; '\
                  + 'cd $curr_dir'
        cmdline = cmdline.format(benchmark_directory,
                                 working_set,
                                 times,
                                 tool,
                                 1 if verify_output else 0,
                                 1 if warmup_cache else 0,
                                 runtime)
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.error('Execute {}'.format(benchmark_directory))
            sys.exit(1)

        if verify_output:
            return IO.load_execution_status(benchmark_directory)
        return True

    @staticmethod
    def cleanup(benchmark_directory,
                compiler):
        """Clean up a directory.

        YaCoS use the tool make to compile a benchmark.
        So, this function invokes Makefile cleanup.

        Parameters
        ----------
        benchmark_directory : str

        compiler : str
            The compiler: clang, llvm, icc, gcc

        """
        cmdline = 'curr_dir=$PWD ; ' \
                  + 'cd {0} ; ' \
                  + 'make -f Makefile.{1} cleanup ; ' \
                  + 'rm -f compile_time.yaml binary_size.yaml ; ' \
                  + 'cd $curr_dir'
        cmdline = cmdline.format(benchmark_directory,
                                 compiler)

        # Invoke the compiler
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.warning('Cleanup: {}'.format(benchmark_directory))

    @staticmethod
    def evaluate(goals,
                 sequence,
                 compiler,
                 benchmark_directory,
                 working_set=0,
                 times=1,
                 tool='hyperfine',
                 verify_output=False,
                 warmup_cache=False,
                 runtime=0,
                 cleanup=True):
        """Compile and execute the benchmark and return the value of the goal.

        Parameters
        ----------
        goals : dict
            {goal: weight}

        sequence(s) : str / dict
            The sequence to compile the benchmark, or the sequences to
            compile each function.

        compiler : str
            The compiler to use.

        benchmark_directory : str

        working_set : int
            The dataset that the benchmark will execute.

        times : int
            Execution times

        tool : str
            Execution tool

        verify_output : bool
            Verify the status of the execution

        warmup_cache: bool

        runtime: int
            The runtime execution (timeout)

        cleanup : bool
            If True cleanup the benchmark directory
        Return
        ------
        goal : float
            The value of the goal.
        """
        # Compile the benchmark N times if compile_time is the goal.
        times_ = times if times > 2 else 3

        if isinstance(sequence, str):
            Engine.compile_(benchmark_directory,
                            compiler,
                            sequence,
                            working_set,
                            times_ if Goals.has_compile_time(goals) else 1)
        elif isinstance(sequence, dict):
            Engine.extract_globals_and_functions(benchmark_directory)
            Engine.compile_functions(benchmark_directory,
                                     compiler,
                                     sequence,
                                     working_set,
                                     times_ if Goals.has_compile_time(goals) else 1)

        if verify_output and Goals.only_compile_time_goal(goals):
            out_ok = Engine.execute(benchmark_directory,
                                    working_set,
                                    1,
                                    tool,
                                    verify_output,
                                    warmup_cache,
                                    runtime)
        elif Goals.has_dynamic_goal(goals):
            out_ok = Engine.execute(benchmark_directory,
                                    working_set,
                                    times_,
                                    tool,
                                    verify_output,
                                    warmup_cache,
                                    runtime)

        if verify_output and not out_ok:
            Engine.cleanup(benchmark_directory,
                           compiler)
            return float('inf')

        goal_value = 0.0
        for goal, weight in goals.items():
            # Extract the value of the goal.
            goal_value += Goals.goal(goal)(benchmark_directory) * weight

        if cleanup:
            Engine.cleanup(benchmark_directory,
                           compiler)

        return goal_value

    @staticmethod
    def only_evaluate(goals,
                      benchmark_directory,
                      working_set=0,
                      times=1,
                      tool='hyperfine',
                      verify_output=False,
                      warmup_cache=False,
                      runtime=0,
                      cleanup=True):
        """Execute the benchmark and return the value of the goal.

        Parameters
        ----------
        goals : dict
            {goal: weight}

        sequence(s) : str / dict
            The sequence to compile the benchmark, or the sequences to
            compile each function.

        benchmark_directory : str

        working_set : int
            The dataset that the benchmark will execute.

        times : int
            Execution times

        tool : str
            Execution tool

        verify_output : bool
            Verify the status of the execution

        warmup_cache: bool

        runtime: int
            The runtime execution (timeout)

        cleanup : bool
            If True cleanup the benchmark directory

        Return
        ------
        goal : float
            The value of the goal.
        """
        times_ = times if times > 2 else 3

        if verify_output and Goals.only_compile_time_goal(goals):
            out_ok = Engine.execute(benchmark_directory,
                                    working_set,
                                    1,
                                    tool,
                                    verify_output,
                                    warmup_cache,
                                    runtime)
        elif Goals.has_dynamic_goal(goals):
            out_ok = Engine.execute(benchmark_directory,
                                    working_set,
                                    times_,
                                    tool,
                                    verify_output,
                                    warmup_cache,
                                    runtime)

        if verify_output and not out_ok:
            Engine.cleanup(benchmark_directory,
                           'opt')
            return float('inf')

        goal_value = 0.0
        for goal, weight in goals.items():
            # Extract the value of the goal.
            goal_value += Goals.goal(goal)(benchmark_directory) * weight

        if cleanup:
            Engine.cleanup(benchmark_directory,
                           'opt')

        return goal_value

    @staticmethod
    def disassemble(benchmark_directory,
                    filename):
        """Disassemble a specific file.

        Parameters
        ----------
        benchmark_directory : str

        filename : str
        """
        cmdline = 'curr_dir=$PWD ; ' \
                  + 'cd {0} ; '\
                  + 'llvm-dis {1}.bc; ' \
                  + 'cd $curr_dir'
        cmdline = cmdline.format(benchmark_directory, filename)

        # Invoke the disassembler
        try:
            subprocess.run(cmdline,
                           shell=True,
                           check=True,
                           capture_output=False)
        except subprocess.CalledProcessError:
            lg.error('Disassemble all: {}'.format(benchmark_directory))
            sys.exit(1)

    @staticmethod
    def extract_globals_and_functions(filename,
                                      benchmark_directory,
                                      global_=True,
                                      llvm=False):
        """Extract variables and functions.

        Parameter
        ---------
        filename : str

        benchmark_directory : str

        variable : bool
            If True generate code for the variables.

        llvm : bool
            If True generate ll code, otherwise bc code.

        Return
        ------
        data : dict
            {'variables': [], 'functions': [],
             'small_variables': {}, 'small_functions': {}}
        """
        driver = LLVMDriver([])
        # Instantiate the LLVM driver.
        # Instantiate the builder.
        builder = R.LLVMNamesBuilder(driver)

        # Extract the number of instructions
        source = os.path.join(benchmark_directory, filename)
        extractionInfo = builder.ir_to_info(source)

        funcs = [data.name for data in extractionInfo.functionInfos]
        globs = [data.name for data in extractionInfo.globalInfos]
        data = {'globals': globs,
                'functions': funcs,
                'small_globals': {},
                'small_functions': {}}

        small_functions = 0
        small_globals = 0
        if global_ and data['globals']:
            for i, item in enumerate(data['globals']):
                global_name = item
                if (global_name[0] == '.'):
                    global_name = global_name[1:]
                    data['globals'][i] = global_name

                if len(global_name) > Engine.file_name_length:
                    name = 'global_{}'.format(small_globals)
                    data['small_globals'][global_name] = name
                    small_globals += 1
                else:
                    name = global_name

                if llvm:
                    cmdline = 'curr_dir=$PWD ; '\
                            + 'cd {0} ; '\
                            + 'llvm-extract --glob={1} -S -o {2}.ll {3} ; '\
                            + 'cd $curr_dir'
                else:
                    cmdline = 'curr_dir=$PWD ; '\
                            + 'cd {0} ; '\
                            + 'llvm-extract --glob={1} -o {2}.bc {3} ; '\
                            + 'cd $curr_dir'
                cmdline = cmdline.format(benchmark_directory,
                                         item,
                                         name,
                                         filename)

                try:
                    subprocess.run(cmdline,
                                   shell=True,
                                   check=True,
                                   capture_output=False)
                except subprocess.CalledProcessError:
                    lg.error('Global: {0}'.format(item))
                    sys.exit(1)

        if data['functions']:
            for item in data['functions']:
                if len(item) > Engine.file_name_length:
                    name = 'function_{}'.format(small_functions)
                    data['small_functions'][item] = name
                    small_functions += 1
                else:
                    name = item

                if llvm:
                    cmdline = 'curr_dir=$PWD ; '\
                            + 'cd {0} ; '\
                            + 'llvm-extract --func={1} -S -o {2}.ll {3} ; '\
                            + 'cd $curr_dir'
                else:
                    cmdline = 'curr_dir=$PWD ; '\
                            + 'cd {0} ; '\
                            + 'llvm-extract --func={1} -o {2}.bc {3} ; '\
                            + 'cd $curr_dir'
                cmdline = cmdline.format(benchmark_directory,
                                         item,
                                         name,
                                         filename)

                try:
                    subprocess.run(cmdline,
                                   shell=True,
                                   check=True,
                                   capture_output=False)
                except subprocess.CalledProcessError:
                    lg.error('Function: {0}'.format(item))
                    sys.exit(1)

        return data

    @staticmethod
    def optimize_functions(benchmark_directory,
                           sequences):
        """Optimize each function.

        Parameter:
        ---------
        benchmark_directory : str

        sequences : dict
            {function: sequence}
        """
        for function, sequence in sequences.items():
            cmdline = 'curr_dir=$PWD; cd {0} ; ' \
                    + 'opt {1} {2}.bc -o {2}.bc ; ' \
                    + 'cd $curr_dir'
            cmdline = cmdline.format(benchmark_directory,
                                     sequence,
                                     function)
            try:
                subprocess.run(cmdline,
                               shell=True,
                               check=True,
                               capture_output=False)
            except subprocess.CalledProcessError:
                lg.error('Optimize: {}'.format(function))
                sys.exit(1)
