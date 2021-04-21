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

import pygmo as pg

from yacos.essentials import Sequence
from yacos.essentials import IO
from yacos.essentials import Engine


class Pygmo:
    """A Pygmo's strategy."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #       'seq': list}}
    __results = None

    # SGA
    # {gen = {'fevals': int,
    #         'best': float,
    #         'improvement': float}}
    #
    # PSO
    # {gen: {'fevals': int,
    #        'gbest': float,
    #        'meanvel': float,
    #        'meanlbest': float,
    #        'avgdist': float}
    __log = None

    class Problem:
        """Pygmo's problem."""

        def __init__(self,
                     first_key,
                     last_key,
                     passes_dict,
                     dimension,
                     goal,
                     compiler,
                     benchmark_directory,
                     working_set,
                     times,
                     tool,
                     verify_output):
            """Construct a Pygmo problem.

            Parameters
            ----------
            first_key : int
                The index of the first pass.

            last_key : int
                The index of the last pass.

            passes_dict : dict
                The dictionary with the available passes.

            dimension : int
                The length of a sequence.

            goal : str

            compiler : str

            benchmark_directory : str

            working_set : int

            times: int

            tool: str
                Execution tool

            verify_output: bool
                The goal is valid only if the execution status is OK.
            """
            self.first_key = first_key
            self.last_key = last_key
            self.passes_dict = passes_dict
            self.dimension = dimension
            self.goal = goal
            self.compiler = compiler
            self.benchmark_directory = benchmark_directory
            self.working_set = working_set
            self.times = times
            self.tool = tool
            self.verify_output = verify_output

        def __deepcopy__(self,
                         *args,
                         **kwargs):
            """Deeep copy."""
            return self

        def fitness(self,
                    sequence):
            """Calculate and return the fitness."""
            sequence = Sequence.fix_index(list(sequence))
            sequence = Sequence.sanitize(sequence)
            sequence = Sequence.index_pass_to_list(sequence,
                                                   self.passes_dict)
            goal_value = Engine.evaluate(self.goal,
                                         Sequence.name_pass_to_string(
                                             sequence
                                         ),
                                         self.compiler,
                                         self.benchmark_directory,
                                         self.working_set,
                                         self.times,
                                         self.tool,
                                         self.verify_output)
            return [goal_value]

        def get_nix(self):
            """Integer dimension of the problem."""
            return self.dimension

        def get_bounds(self):
            """Box-bounds."""
            return ([self.first_key] * self.dimension,
                    [self.last_key] * self.dimension)

        def get_name(self):
            """Problem name."""
            return 'Optimization Selection'

        def get_extra_info(self):
            """Info."""
            return '\tDimensions: ' + str(self.dimension)

    @dataclass
    class PygmoFlags:
        """Pygmo flags.

        Parameters
        ----------
        first_key : int
            The index of the first pass.

        last_key : int
            The index of the last pass.

        passes_dict : dict
            The dictionary with the available passes.

        dimension : int
            The length of a sequence.

        population : int

        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times: int
            Execution times

        tool : str
            Execution tool

        verify_output: bool
            The goal is valid only if the execution status is OK.
        """

        first_key: int
        last_key: int
        passes_dict: dict
        dimension: int
        population: int
        goals: dict
        compiler: str
        benchmarks_directory: str
        working_set: int
        times: int
        tool: str
        verify_output: bool

    def __init__(self,
                 dimension,
                 population,
                 passes_filename,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize the arguments.

        Parameters
        ----------
        dimension : int
            The length of a sequence.

        population : int

        passes_filename : str
            The file that describes the passes to use.

        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times: int
            Execution times

        tool: str
            Execution tool

        verify_output: bool
            The goal is valid only if the execution status is OK.
        """
        first_key, last_key, passes_dict = IO.load_passes(passes_filename)

        # When the goal is obtained during compile time
        # and the working set is not defined during compilation,
        # we do not need the working set.
        self.__flags = self.PygmoFlags(first_key,
                                       last_key,
                                       passes_dict,
                                       dimension,
                                       population,
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

    @property
    def log(self):
        """Getter."""
        return self.__log

    def exec(self, algorithm, benchmark):
        """Execute the algorithm.

        Parameter
        ---------
        algorithm : Pygmo algorithm

        benchmark : str
        """
        # Step 1: Algorithm
        algorithm = pg.algorithm(algorithm)
        # algorithm.set_verbosity(1)

        # Step 2: Instantiate a pygmo problem
        index = benchmark.find('.')

        # Benchmark directtory
        bench_dir = os.path.join(self.__flags.benchmarks_directory,
                                 benchmark[:index],
                                 benchmark[index+1:])

        problem = self.Problem(self.__flags.first_key,
                               self.__flags.last_key,
                               self.__flags.passes_dict,
                               self.__flags.dimension,
                               self.__flags.goals,
                               self.__flags.compiler,
                               bench_dir,
                               self.__flags.working_set,
                               self.__flags.times,
                               self.__flags.tool,
                               self.__flags.verify_output)
        problem = pg.problem(problem)

        # Step 3: The initial population
        population = pg.population(problem,
                                   self.__flags.population)

        # Step 4: Evolve the population
        population = algorithm.evolve(population)

        # Step 5: Get the results
        sga_sequence = population.get_x().tolist()
        sga_fitness = population.get_f().tolist()

        self.__results = {}
        for index in range(self.__flags.population):
            sequence = Sequence.index_pass_to_list(sga_sequence[index],
                                                   self.__flags.passes_dict)

            goal_value = sga_fitness[index][0]

            if goal_value == float('inf'):
                continue

            self.__results[index] = {'seq': sequence,
                                     'goal': goal_value}

        # Step 6: Get the log
        self.__log = {}
        if algorithm.get_name() == 'SGA: Genetic Algorithm':
            uda = algorithm.extract(pg.sga)
            log = uda.get_log()
            for (gen, fevals, best, improvement) in log:
                self.__log[gen] = {'fevals': fevals,
                                   'best': best,
                                   'improvement': improvement}
        elif algorithm.get_name() == 'PSO: Particle Swarm Optimization':
            uda = algorithm.extract(pg.pso)
            log = uda.get_log()
            for (gen, fevals, gbest, meanvel, meanlbest, avgdist) in log:
                self.__log[gen] = {'fevals': fevals,
                                   'gbest': gbest,
                                   'meanvel': meanvel,
                                   'meanlbest': meanlbest,
                                   'avgdist': avgdist}


class SGA(Pygmo):
    """Simple Genetic Algorithm."""

    __version__ = '1.0.0'

    __flags = None

    @dataclass
    class Flags:
        """Pygmo flags.

        Parameters
        ----------
        generations : int

        cr : float
            Crossover probability

        m : float
            Mutation probability

        param_m  : float
            Distribution index (polynomial mutation),
            gaussian width (gaussian mutation) or
            inactive (uniform mutation)

        param_s : float
            The number of best individuals to use in “truncated”
            selection or the size of the tournament in
            tournament selection.

        crossover : str
            exponential, binomial or single

        mutation : str
            gaussian, polynomial or uniform

        selection : str
            tournament or truncated

        seed : int
        """

        generations: int
        cr: float
        m: float
        param_m: float
        param_s: float
        crossover: str
        mutation: str
        selection: str
        seed: int

    def __init__(self,
                 generations,
                 population,
                 cr,
                 m,
                 param_m,
                 param_s,
                 crossover,
                 mutation,
                 selection,
                 seed,
                 dimension,
                 passes_filename,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize a SGA object.

        Parameters
        ----------
        generations : int

        population : int

        cr : float
            Crossover probability

        m : float
            Mutation probability

        param_m  : float
            Distribution index (polynomial mutation),
            gaussian width (gaussian mutation) or
            inactive (uniform mutation)

        param_s : float
            The number of best individuals to use in “truncated”
            selection or the size of the tournament in
            tournament selection.

        crossover : str
            exponential, binomial or single

        mutation : str
            gaussian, polynomial or uniform

        selection : str
            tournament or truncated

        seed : int

        dimension : int
            The length of a sequence.

        passes_filename : str
            The file that describes the passes to use.

        goals : dict

        compiler : str

        benchmarks_directory : str

        working_set : int
            The dataset to execute the benchmark.

        times : int
            Execution times

        tool : str
            Execution tool

        verify_output: bool
            The goal is valid only if the execution status is OK.
        """
        self.__flags = self.Flags(generations,
                                  cr,
                                  m,
                                  param_m,
                                  param_s,
                                  crossover,
                                  mutation,
                                  selection,
                                  seed)
        super().__init__(dimension,
                         population,
                         passes_filename,
                         goals,
                         compiler,
                         benchmarks_directory,
                         working_set,
                         times,
                         tool,
                         verify_output)

    def run(self, benchmark):
        """Execute the algorithm.

        Parameter
        --------
        benchmark: str
        """
        if self.__flags.seed is None:
            algorithm = pg.sga(gen=self.__flags.generations,
                               cr=self.__flags.cr,
                               m=self.__flags.m,
                               param_m=self.__flags.param_m,
                               param_s=self.__flags.param_s,
                               crossover=self.__flags.crossover,
                               mutation=self.__flags.mutation,
                               selection=self.__flags.selection)
        else:
            algorithm = pg.sga(gen=self.__flags.generations,
                               cr=self.__flags.cr,
                               m=self.__flags.m,
                               param_m=self.__flags.param_m,
                               param_s=self.__flags.param_s,
                               crossover=self.__flags.crossover,
                               mutation=self.__flags.mutation,
                               selection=self.__flags.selection,
                               seed=self.__flags.seed)

        # Execute
        super().exec(algorithm, benchmark)


class PSO(Pygmo):
    """Particle Swarm Optimization."""

    __version__ = '1.0.0'

    __flags = None

    @dataclass
    class Flags:
        """PSO flags.

        Parameters
        ----------
        generations : int

        omega : float
            Inertia weight (or constriction factor)

        eta1 : float
            Social component

        eta2 : float
            Cognitive component

        max_vel : float
            Maximum allowed particle velocities
            (normalized with respect to the bounds width)

        variant : int
            Algorithmic variant

        neighb_type : int
            Swarm topology (defining each particle’s neighbours)

        neighb_param : int
            Topology parameter (defines how many neighbours to consider)

        memory : bool
            When true the velocities are not reset between successive
            calls to the evolve method

        seed : int
            Seed used by the internal random number generator.
        """

        generations: int
        omega: float
        eta1: float
        eta2: float
        max_vel: float
        variant: int
        neighb_type: int
        neighb_param: int
        memory: bool
        seed: int

    def __init__(self,
                 generations,
                 population,
                 omega,
                 eta1,
                 eta2,
                 max_vel,
                 variant,
                 neighb_type,
                 neighb_param,
                 memory,
                 seed,
                 dimension,
                 passes_filename,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Initialize a PSO object.

        Parameters
        ----------
        generations : int

        population : int

        omega : float
            Inertia weight (or constriction factor)

        eta1 : float
            Social component

        eta2 : float
            Cognitive component

        max_vel : float
            Maximum allowed particle velocities
            (normalized with respect to the bounds width)

        variant : int
            Algorithmic variant

        neighb_type : int
            Swarm topology (defining each particle’s neighbours)

        neighb_param : int
            Topology parameter (defines how many neighbours to consider)

        memory : bool
            When true the velocities are not reset between successive
            calls to the evolve method

        seed : int
            Seed used by the internal random number generator.
        """
        self.__flags = self.Flags(generations,
                                  omega,
                                  eta1,
                                  eta2,
                                  max_vel,
                                  variant,
                                  neighb_type,
                                  neighb_param,
                                  memory,
                                  seed)

        super().__init__(dimension,
                         population,
                         passes_filename,
                         goals,
                         compiler,
                         benchmarks_directory,
                         working_set,
                         times,
                         tool,
                         verify_output)

    def run(self, benchmark):
        """Execute the algorithm.

        Parameter
        --------
        benchmark : str
        """
        if self.__flags.seed:
            algorithm = pg.pso(self.__flags.generations,
                               self.__flags.omega,
                               self.__flags.eta1,
                               self.__flags.eta2,
                               self.__flags.max_vel,
                               self.__flags.variant,
                               self.__flags.neighb_type,
                               self.__flags.neighb_param,
                               self.__flags.memory,
                               self.__flags.seed)
        else:
            algorithm = pg.pso(self.__flags.generations,
                               self.__flags.omega,
                               self.__flags.eta1,
                               self.__flags.eta2,
                               self.__flags.max_vel,
                               self.__flags.variant,
                               self.__flags.neighb_type,
                               self.__flags.neighb_param,
                               self.__flags.memory)

        # Execute
        super().exec(algorithm, benchmark)
