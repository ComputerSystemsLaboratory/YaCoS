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

import sys
import random as rn
import subprocess

from absl import logging as lg
from yacos.essentials import IO


class Sequence:
    """Static class that manipulates sequences."""

    __version__ = '1.0.0'

    @staticmethod
    def __duplicates(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    @staticmethod
    def update(sequence):
        """Update the sequence.

        LLVM documentation suggests that some optimizations
        could precede and succeed an optimization for its
        effectiveness. Thus, this method add/change a sequence
        to handle such suggestions.

        Look:

        Suresh Purini and Lakshya Jain.
        Finding Good Optimization Sequences Covering Program Space.
        TACO.
        2013

        Parameter
        ----------
        sequence : list
            The original sequence (a list of optimizations).

        Return
        ------
        sequence : list
            The updated sequence.
        """
        newseq = []
        firstime = True
        for _, opt in enumerate(sequence):
            # Predecessor constraints
            if opt in ['-constprop',
                       '-correlated-propagation',
                       '-gvn',
                       '-ipconstprop',
                       '-ipsccp']:
                newseq.append('-reassociate')
            elif opt == '-licm':
                newseq.append('-reassociate')
                newseq.append('-loop-simplify')
            elif opt == '-loop-unswitch':
                newseq.append('-reassociate')
                newseq.append('-loop-simplify')
                newseq.append('-licm')
            elif '-loop-' in opt:
                newseq.append('-indvars')
                newseq.append('-lcssa')
            if opt == '-loop-unroll':
                newseq.append('-loop-rotate')
                if firstime:
                    newseq.append('-unroll-allow-partial')
                    firstime = False
            # Add the actual optimization
            newseq.append(opt)
            # Sucessor Constraints
            if opt == '-tailduplicate':
                newseq.append('-simplifycfg')
            elif opt == '-sccp':
                newseq.append('-dce')
            elif opt in ['-ipconstprop',
                         '-ipsccp']:
                newseq.append('-deadargelim')
                newseq.append('-adce')
            elif opt in ['-constprop']:
                newseq.append('-die')

        return newseq

    @staticmethod
    def sanitize(sequence):
        """Remove consecutive equal optimizations.

        Example
        -------
        [-gvn, -inline, -inline, -sroa] -> [-gvn, -inline, -sroa]

        Such situation can occur due to the generation of random
        elements to compose a sequence.

        Parameter
        ---------
        sequence : list
            The original sequence.

        Return
        ------
        sequence : list
            The clean sequence.
        """
        newseq = sequence[:]
        if len(newseq) == 1:
            return newseq
        i = 0
        while True:
            if i == len(newseq)-1:
                break
            if newseq[i] == newseq[i+1]:
                del newseq[i]
            else:
                i += 1
        return newseq

    @staticmethod
    def mem2reg_first(sequence):
        """-mem2reg should be the first pass.

        Parameter
        ---------
        sequence : list
            The original sequence.

        Return
        ------
        sequence : list
            The new sequence.
        """
        new_sequence = [seq for seq in sequence if seq != '-mem2reg']
        return ['-mem2reg'] + new_sequence

    @staticmethod
    def index_pass_to_list(sequence_index,
                           passes):
        """Convert a index sequence into a pass sequence.

        Example
        -------
        [1, 2, ...] -> [-basicaa, -early-cse, ...]

        Parameter
        ---------
        sequence_index : list
            The sequence composed by pass index.

        passes : dict
            The dictionary which describes the avaliable passes.

        Return
        ------
        sequence : list
            The sequence composed by pass name.
        """
        sequence = [passes[int(round(item, 0))]['pass']
                    for item in sequence_index if item]
        return sequence

    @staticmethod
    def index_pass_to_string(sequence_index,
                             passes):
        """Convert a index sequence into a string.

        Example
        -------
        [1, 2, ...] -> "-basicaa -early-cse ..."

        Parameter
        ---------
        sequence_index : list
            The sequence composed by pass index.

        passes : dict
            The dictionary which describes the avaliable passes.

        Return
        ------
        sequence : str
            The sequence composed by pass name.
        """
        sequence = ' '.join([passes[int(round(item, 0))]['pass']
                            for item in sequence_index if item])
        return sequence

    @staticmethod
    def remove_passes(sequence,
                      bool_vector):
        """Turn off (remove) some passes.

        Parameters
        ----------
        sequence : list
            The original sequence

        bool_vector : list
            The list that indicates which passes to turn off.

        Return
        ------
        sequence: list
            The new sequence.
        """
        new_sequence = [seq for i, seq in enumerate(sequence)
                        if bool_vector[i]]
        return new_sequence

    @staticmethod
    def name_pass_to_string(sequence):
        """Convert a list of names to string.

        Example
        -------
        [-basicaa, -early-cse, ...] -> "-basicaa -early-cse ..."

        Parameter
        ---------
        sequence : list
            A list of passes.

        Return
        ------
        sequence : str
            A string.
        """
        string = ' '.join(sequence)
        return string

    @staticmethod
    def string_to_name_pass(sequence):
        """Convert string to a list of names.

        Example
        -------
        "-basicaa -early-cse ..." -> [-basicaa, -early-cse, ...]

        Parameter
        ---------
        sequence : string
            A sequence of passes.

        Return
        ------
        sequence : list
            A sequence.
        """
        lst_sequence = sequence.split()
        return lst_sequence

    @staticmethod
    def fix_index(sequence):
        """Fix the pass index (float to int).

        Some Pygmo's algorithms generate float numbers (index of the
        pass), so we need to convert them to int.

        Parameter
        ---------
        sequence : list
            The sequence.

        Return
        ------
        sequence : list
            The new sequence.
        """
        new_seq = [int(round(pass_)) for pass_ in sequence]
        return new_seq

    @staticmethod
    def get_the_best(sequences,
                     nof_sequences=1):
        """Return the best sequences based on the goal value.
           This method removes ties based on lenght (the smallest the best).

        Parameters
        ----------
        sequences : dict
            The dictionary with the sequences.

        nof_sequence : int
            The number of sequences to extract from sequences.

        Return
        ------
        best : dict
            The N best sequences.
            {key: list}
        """
        if len(sequences) < nof_sequences:
            lg.error('There is no {} sequences '.format(nof_sequences))
            sys.exit(1)

        goals = [data['goal'] for _, data in sequences.items()]
        keys = [key for key, _ in sequences.items()]

        duplicates = dict((x, Sequence.__duplicates(goals, x))
                          for x in set(goals) if goals.count(x) > 1)

        remove = []
        for _, positions in duplicates.items():
            rank = [(len(sequences[keys[pos]]['seq']), keys[pos])
                    for pos in positions]
            rank.sort(reverse=True)
            del rank[-1]
            for _, key in rank:
                remove.append(key)

        sequences_ = sequences.copy()
        for key in remove:
            del(sequences_[key])

        rank = [(data['goal'], key) for key, data in sequences_.items()]
        rank.sort()

        best = {}
        for i in range(nof_sequences):
            if i < len(rank):
                key = rank[i][1]
                best[key] = sequences[key].copy()
        return best

    @staticmethod
    def get_all_best(sequences):
        """Return all best sequences based on the goal value.
           This method returns all best sequences, this means the K
           sequences with the same goal value.

        Parameters
        ----------
        sequences : dict
            The dictionary with the sequences.

        Return
        ------
        best : dict
            The N best sequences.
            {key: list}
        """
        if len(sequences) < 1:
            lg.error('There is no sequences')
            sys.exit(1)

        rank = [(data['goal'], key) for key, data in sequences.items()]
        rank.sort()

        best = {}
        best_goal = rank[0][0]
        for i, (goal, key) in enumerate(rank):
            if best_goal == goal:
                key = rank[i][1]
                best[key] = sequences[key].copy()
            else:
                break
        return best

    @staticmethod
    def filter(sequences,
               value):
        """Return the sequences whose goal is better than "value".

        Parameters
        ----------
        sequences : dict
            The dictionary with the sequences.

        value : int
            The threshold

        Return
        ------
        best : dict
            The N best sequences.
            {key: list}
        """
        if len(sequences) < 1:
            lg.error('There is no sequences')
            sys.exit(1)

        best = {}
        for key, seq_data in sequences.items():
            if seq_data['goal'] <= value:
                best[key] = sequences[key].copy()
        return best

    @staticmethod
    def get_the_smallest(sequences):
        """Return the smallest sequences.

        Parameters
        ----------
        sequences : dict
            The dictionary with the sequences.

        Return
        ------
        small : list
            The smallest sequence.
        """
        rank = [(len(data['seq']), key) for key, data in sequences.items()]
        return sequences[rank[0][1]]

    @staticmethod
    def exist(sequence,
              sequences):
        """Verify if a sequence exists.

        Parameters
        ----------
        sequence : list
            The sequence to verify.

        sequences : dict
            The dictionary with the sequences.

        Return
        ------
        exist : bool
            True if the sequence exists in the dictionary, otherwise False.
        """
        find = False
        for _, data in sequences.items():
            if data['seq'] == sequence:
                find = True
                break
        return find

    @staticmethod
    def create_random_sequence(first_key,
                               last_key,
                               passes,
                               minimum,
                               maximum,
                               repetition):
        """Create a sequence.

        Parameters
        ----------
        first_key : int
            The index of the first pass.

        last_key : int
            The index of the last pass.

        passes : dict
            The dictionary with the available passes to use.

        mininum : int
            The minimum length of the sequence.

        maximum : int
            The maximum length of the sequence.

        repetition : bool
            Enable repetition ?

        Return
        ------
        sequence : list
            A list of optimization (a sequence).
        """
        sequence = []
        if first_key == 0:
            first_key = 1
        if minimum != maximum:
            real_length = rn.randint(minimum, maximum)
        else:
            real_length = max

        while True:
            pass_ = rn.randint(first_key, last_key)
            insert = True
            if (not repetition) and (pass_ in sequence):
                insert = False
            if insert:
                sequence.append(pass_)
            if len(sequence) == real_length:
                break

        sequence = Sequence.index_pass_to_list(sequence,
                                               passes)

        return sequence

    @staticmethod
    def create_random_sequences(nof_sequences,
                                minimum,
                                maximum,
                                factor,
                                ssa,
                                shuffle,
                                update_,
                                repetition,
                                original,
                                passes_filename):
        """Create N random sequences.

        Parameters
        ----------
        nof_sequences : int
            The number of sequences.

        minimum : int
            The minimum and maximum length of the sequence.

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

        Return
        ------
        sequences : dict
            A dictionary which contains N random sequences.
        """
        if (not repetition) and (maximum > (maximum*0.7)):
            lg.error('adjust MAXIMUM lenght. MAXIMUM \
            should be less than 70% of |PASSES|')
            sys.exit(1)
        if not (original or update_ or shuffle):
            lg.error('Error: it is necessary to use at \
            least one argument (-original, -update, -shuffle)')
            sys.exit(1)

        # Load the passes
        first_key, last_key, passes = IO.load_passes(passes_filename)
        counter = 0
        sequences = {}
        nof_sequences *= factor

        while True:
            # generate a sequence
            seq = Sequence.create_random_sequence(first_key,
                                                  last_key,
                                                  passes,
                                                  minimum,
                                                  maximum,
                                                  repetition)
            seq = Sequence.sanitize(seq)

            if ssa:
                seq = Sequence.mem2reg_first(seq)

            if original:
                if not Sequence.exist(seq,
                                      sequences):
                    sequences[counter] = {'seq': seq}
                    counter += 1
                    if counter >= nof_sequences:
                        break
                if shuffle:
                    sseq = seq[:]
                    rn.shuffle(sseq)
                    sseq = Sequence.sanitize(sseq)
                    if not Sequence.exist(sseq,
                                          sequences):
                        sequences[counter] = {'seq': sseq}
                        counter += 1
                        if counter >= nof_sequences:
                            break
            if update_:
                seq = Sequence.update(seq)
                seq = Sequence.sanitize(seq)

                if not Sequence.exist(seq,
                                      sequences):
                    sequences[counter] = {'seq': seq}
                    counter += 1
                    if counter >= nof_sequences:
                        break

                if shuffle:
                    seq = Sequence.update(seq)
                    seq = Sequence.sanitize(seq)
                    if not Sequence.exist(seq,
                                          sequences):
                        sequences[counter] = {'seq': seq}
                        counter += 1
                        if counter >= nof_sequences:
                            break

        return sequences

    @staticmethod
    def expand(passes):
        """Expand the sequence.
           LLVM adds several passes in the sequence.

           Example:
           passes: -constprop

           LLVM invokes: -targetlibinfo -tti -constprop -verify

           See:
           opt --disable-output <sequence> --debug-pass=Arguments < /dev/null

        Argument
        --------
        passes : list

        Return
        ------
        sequences : list
        """
        str_passes = Sequence.name_pass_to_string(passes)
        cmdline = 'opt --disable-output {}'.format(str_passes) \
                  + ' --debug-pass=Arguments < /dev/null'

        # Invoke the compiler
        try:
            ret = subprocess.run(cmdline,
                                 shell=True,
                                 check=True,
                                 capture_output=True)
        except subprocess.CalledProcessError:
            lg.warning('Expand: {}'.format(passes))
            return ''

        expand_passes = ret.stderr.decode()
        expand_passes = expand_passes.replace('\n', '')
        expand_passes = expand_passes.replace('Pass Arguments:  ', '')

        # Return
        return Sequence.string_to_name_pass(expand_passes)

    @staticmethod
    def split(passes,
              expand=False):
        """Generate small sequences.

        Argument
        --------
        passes : list

        expand : bool
            If True evaluate the expanded sequence.

        Return
        ------
        sequences : dict
        """
        new_passes = passes[:]
        if expand:
            new_passes = Sequence.expand(passes)

        sequences = {}
        for i in range(1, len(new_passes)+1):
            sequences[i-1] = {'seq': new_passes[:i]}
        return sequences
