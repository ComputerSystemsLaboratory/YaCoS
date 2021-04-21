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
import yaml as yl
import pickle as pk
from absl import logging as lg


class IO:
    """Static class that manipulates files."""

    __version__ = '1.0.0'

    @staticmethod
    def load_yaml(filename):
        """Load an yaml file.

        Parameter
        --------
        filename: str

        Return
        ------
        data : dict or list
        """
        # Load the file
        try:
            fin = open(filename, 'r')
            data = yl.safe_load(fin)
        except IOError:
            return False
        except yl.YAMLError:
            fin.close()
            return False
        fin.close()

        # Return the data
        return data

    @staticmethod
    def load_yaml_or_fail(filename):
        """Load an yaml file.

        Parameter
        --------
        filename: str

        Return
        ------
        data : dict or list
        """
        # Verify whether the filename exists or not.
        # If not, abort the execution.
        if not os.path.isfile(filename):
            lg.error('The file {} does not exit'.format(filename))
            sys.exit(1)

        # Load the file
        fin = open(filename, 'r')
        try:
            data = yl.safe_load(fin)
        except yl.YAMLError:
            fin.close()
            lg.error('Yaml error.')
            sys.exit(1)
        fin.close()

        # Return the data
        return data

    @staticmethod
    def dump_yaml(data,
                  filename,
                  directory='',
                  only_the_best=False):
        """Store yaml data.

        Parameters
        ----------
        data : dict

        filename : str

        directory : str
            The directory to store the report.

        only_the_best : bool
            If true, store only the best result. In this
            case, we should verify the goal value.
        """
        if directory:
            output = '{}/{}'.format(directory,
                                    filename)
        else:
            output = filename

        if only_the_best:
            rank = []
            for index, value in data.items():
                rank.append((value['goal'], index))
            rank.sort()
            index = rank[0][1]
            data = {0: data[index]}

        fout = open(output, 'w')

        yl.dump(data,
                fout,
                default_flow_style=False)

        fout.close()

    @staticmethod
    def load_pickle(filename):
        """Load a pickle file.

        Parameter
        --------
        filename: str

        Return
        ------
        data : Any
        """
        # Load the file
        try:
            fin = open(filename, 'rb')
            data = pk.load(fin)
        except IOError:
            return False
        except pk.PickleError:
            fin.close()
            return False
        fin.close()

        # Return the data
        return data

    @staticmethod
    def load_pickle_or_fail(filename):
        """Load an yaml file.

        Parameter
        --------
        filename: str

        Return
        ------
        data : Any
        """
        # Verify whether the filename exists or not.
        # If not, abort the execution.
        if not os.path.isfile(filename):
            lg.error('The file {} does not exit'.format(filename))
            sys.exit(1)

        # Load the file
        fin = open(filename, 'rb')
        try:
            data = pk.load(fin)
        except pk.PickleError:
            fin.close()
            lg.error('Pickle error.')
            sys.exit(1)
        fin.close()

        # Return the data
        return data

    @staticmethod
    def dump_pickle(data,
                    filename,
                    directory=''):
        """Dump pickle data.

        Parameters
        ----------
        data : Any

        filename : str

        directory : str
            The directory to store the report.
        """
        if directory:
            output = '{}/{}'.format(directory,
                                    filename)
        else:
            output = filename

        fout = open(output, 'wb')
        pk.dump(data, fout)
        fout.close()

    @staticmethod
    def read_lines(filename):
        """Read the lines of a file.

        Parameter
        ---------
        filename : str

        Return
        ------
        lines : list
        """
        # Verify whether the filename exists or not.
        # If not, abort the execution.
        if not os.path.isfile(filename):
            lg.error('The file {} does not exit'.format(filename))
            sys.exit(1)

        # Load the file
        fin = open(filename, 'r')
        try:
            lines = fin.readlines()
        except IOError:
            fin.close()
            lg.error('It is not possible to read the file {}'.format(filename))
            sys.exit(1)
        fin.close()

        # Return the lines
        return lines

    @staticmethod
    def load_passes(filename):
        """Load the yaml file, which stores the passes.

        Parameter
        ---------
        filename : str

        Returns
        -------
        first_key : int
            The index of the first pass.

        last_key : int
            The index of the last pass.

        passes : dict
            The dictionary with the available passes.
        """
        passes_dict = IO.load_yaml_or_fail(filename=filename)
        passes_keys = passes_dict.keys()
        num_passes = len(passes_keys)
        first_key = min(passes_dict.keys())
        last_key = max(passes_dict.keys())

        # Validate the keys
        none_pass = False
        for key in passes_dict.keys():
            if passes_dict[key]['pass'] == 'NONE':
                if key == 0:
                    none_pass = True
                else:
                    lg.error('Ivalid passes file ({}), NONE should be \
                    the key Zero'.format(filename))
                    sys.exit(1)

        first_item = 0 if none_pass else first_key
        last_item = first_item + (num_passes-1) * 1
        arith_progression = (num_passes * (first_item + last_item))/1
        arith_progression_found = (num_passes * (first_key + last_key))/1

        if arith_progression != arith_progression_found:
            lg.error('Invalid passes file ({}), the keys should be \
            numeric and continuous'.format(filename))
            sys.exit(1)

        # Returns
        return first_key, last_key, passes_dict

    @staticmethod
    def load_execution_status(benchmark_directory):
        """Load the execution status.

        Parameter
        --------
        benchmark_directory : str

        Return
        ------
        True if execution ok, otherwise False
        """
        data = IO.load_yaml(
            '{}/verify_output.yaml'.format(benchmark_directory)
        )
        return data == 'succeed'
