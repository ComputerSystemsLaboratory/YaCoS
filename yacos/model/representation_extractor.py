import os

from yacos.essential import Engine
from yacos.info.ncc import Inst2Vec

class RepresentationExtractor:

    @staticmethod
    def process_inst2vec(benchmark_dir, 
                         compile_system='opt', 
                         opt_set='-O0'):
        """
        Static Method to get inst2vec features from benchmark
        Parameters
        ----------
        benchmark_dir: str 
            The directory that benchmark is stored into file system
        compile_system: str
            The compile system that will be used to build the IR and get representation
            The directory of benchmark must have a Makefile.compile_system (e.g. Makefile.opt)
            and compile.sh script
        opt_set: str
            The optimization set that will be used to extract the representation

        Return
        ------
         The inst2vec matrix
        """
        Engine.compile(benchmark_dir, compile_system, opt_set)
        Engine.disassemble(benchmark_dir, 'a.out_o')

        spl = os.path.split(benchmark_dir)
        while (spl[1] == ''):
            spl = os.path.split(spl[0])
        bench_name = spl[1]
        spl2 = os.path.split(spl[0])
        while (spl2[1] == ''):
            spl2 = os.path.split(spl2[0])
        benchmarks_directory = spl2[0]
        bench_suite = spl2[1]

        benchmark = '.'.join([bench_suite,bench_name])
        Inst2Vec.prepare_benchmark(benchmark, benchmarks_directory)
        #print(benchmark)
        rep = Inst2Vec.extract()
        values = list(rep.values())
        values = values[0]

        Engine.cleanup(benchmark_dir, compile_system)

        Inst2Vec.remove_data_directory()

        return values

    @staticmethod
    def get_inst2vec_features(directory,
                              compile_system='opt', 
                              opt_set='-O0'):
        vec = RepresentationExtractor.process_inst2vec(directory,
                                                       compile_system,
                                                       opt_set)
        acc_col = vec.sum(axis=0)
        rep = acc_col.tolist()[0]
        return rep