"""
Copyright 2020 Alexander Brauckmann.

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

from absl import logging as lg

from yacos.info.compy.llvm_seq import LLVMSeqBuilder


program_1fn_2 = """
int bar(int a) {
  if (a > 10)
    return a;
  return -1;
}
"""


program_fib = """
int fib(int x) {
    switch(x) {
        case 0:
            return 0;
        case 1:
            return 1;
        default:
            return fib(x-1) + fib(x-2);
    }
}
"""


def verify_data_dir():
    top_dir = os.path.join(os.environ.get('HOME'), '.local')
    if not os.path.isdir(os.path.join(top_dir, 'yacos')):
        lg.error('YaCoS data does not exist.')
        sys.exit(1)


def test_construct_with_custom_visitor():
    """Construction."""
    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_1fn_2.c')
    builder = LLVMSeqBuilder()
    info = builder.source_to_info(filename)
    _ = builder.info_to_representation(info)


def test_plot(tmpdir):
    """General tests: Plot."""
    verify_data_dir()
    filename = os.path.join(os.environ.get('HOME'),
                            '.local',
                            'yacos',
                            'tests',
                            'program_fib.c')
    builder = LLVMSeqBuilder()
    info = builder.source_to_info(filename)
    seq = builder.info_to_representation(info)

    outfile = os.path.join(tmpdir, "syntax_seq.png")
    seq.draw(path=outfile, width=8)

    assert os.path.isfile(outfile)
