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
import copy

from dataclasses import dataclass
from typing import Callable
from networkx import algorithms
from networkx.algorithms import traversal

from yacos.essential import Sequence
from yacos.essential import Engine
from yacos.essential import IO
from yacos.essential import Goal

from yacos.info.ncc import Inst2Vec

from yacos.essential import Similarity

from yacos.model import NetModel


class GraphFromSequences:
    """Graph From Sequences.

    A Graph-based Model for Building Optimization Sequences
    A Study Case on Code Size Reduction

    Nilton and Anderson
    SBLP - 2021
    """

    __version__ = '2.1.0'

    __flags = None

    

    # {key: {'goal': float,
    #        'seq': list}}
    __results = None

    @dataclass
    class Flags:
        """ GraphFromSequences flags.
            See __init__ method of GraphFromSequences class
            documentation to understand what each flag means.
        """
        netmodel_file: str
        benchmarks_representation_file: str

    def __init__(self,  
                 netmodel_file,
                 benchmarks_representation_file):
        """Initialize a GraphFromSequences object.
            
            netmodel_file: str
                File name where the NetModel will be saved
            
            benchmarks_representation_file:
                File name where the benchmarks representations will be saved


            
        """
        self.__flags = self.Flags(netmodel_file,
                                  benchmarks_representation_file)
        self.__Graph_Model = None


    @staticmethod
    def create_file_list(train_bench_base_dir, 
                         perf_dict_base_dir,
                         yaml_file,
                         representation_func):
        """
            Create a list to insert the objects into NetModel
            Parameters
            ----------

            train_bench_base_dir: str
                The directory where benchamrks collections are stored

            yaml_file: str
                The name of an YAML file containing benchmarks used to build the model
                    it must be a list, where each element is 
                        BenchCollection.BenchName
                    the filesystem must have a directory named 
                        BenchCollection, that is inside benchmark_train_directory
                    and inside of benchmark_train_directory/BenchCollection
                    must have a directory named 
                        BenchName 
                    Finally, inside benchmark_train_directory/BenchCollection/BenchName
                        must have the makefile to build the benchmark
            
            perf_dict_base_dir: string
                The directory where the YAML files containing the partial compilation
                are store.
                The structure also use benchmark_set file.
                Each YAML must be stored as:
                perf_dict_base_dir/BenchCollection/BenchName.yaml
            
            representation_func: function(str) -> representation
                the function used to extract benchmarks representation
                    str: is a string where the benchmark source code is stored 
                    to exctract representation
                    function: fuction used to exctract the representation of
                    the train benchmark

            Returns
            -------
            a list of tuples 
            each tuple is composed by (benchmark_name, performance_dictionary, representation)
                benchmark_name: str
                performance_dictionary: dictionary
                    see it keys and values at add_sequence method (NetModel class in net_model.py file)
                representation: not defined
                    This is the representation of benchmark (e.g. inst2vec, msf, etc)
                    The representation doesn't need to have a specific type, it will be stored
                    into a pickle file.

            The returned strucuture is used in build method of this class
        """
        all_names=[]
        yaml_names = IO.load_yaml(yaml_file)
        if yaml_names != False:
            lst=[]
            for name in yaml_names:
                print(name)
                idx = name.find('.')
                bench_collection = name[:idx]
                bench_name = name[idx+1:]

                bench_dir = os.path.join(train_bench_base_dir, 
                                         bench_collection,
                                         bench_name)
                
                representation = representation_func(bench_dir)
                bench_filename = bench_name+'.yaml'

                f = os.path.join(perf_dict_base_dir,
                                 bench_collection,
                                 bench_filename)
                d = IO.load_yaml(f)
                if d != False:
                    #essa linha deve ser adicionada nos arquivos novos(code size)
                    #pois antes das sequencias parciais tem uma chave "aleatoria"
                    d=d[list(d.keys())[0]]
                    lst.append((bench_name,d,representation))
                else:
                    print('============WARNING==========')
                    print('File not found: ',f)
                    print('Data will be built without file:', f)
                    print('=============================')
        else:
            print('error opening: ',yaml_file)
            exit()
        return lst

    def build(self,
              file_list):
        """Build the graph (model)."""
        self.__Graph_Model = NetModel()
        self.__Graph_Model.insert_sequences(file_list)

    def load(self):
        """Load the model"""
        if self.__Graph_Model == None:
            self.__Graph_Model = NetModel()

        self.__Graph_Model.load_net(self.__flags.netmodel_file,
                                    self.__flags.benchmarks_representation_file)
        

    def save(self):
        """Save the model"""
        self.__Graph_Model.save_net(self.__flags.netmodel_file, 
                                    self.__flags.benchmarks_representation_file)

    def get_graph(self):
        return self.__Graph_Model.graph

    def get_initial_opt(self,program):
        return self.__Graph_Model.initial_nodes[program]

    def get_representations(self):
        return self.__Graph_Model.representations

    def __neightbors_similarity(self,
                                graph,
                                opt,
                                sim_table,
                                edge_weight,
                                node_weight,
                                node_func,
                                edge_func):
        neighbors_sim = {}
        for u in graph.adj[opt]:
            u_neigh = graph.nodes[u]['programs']
            u_sims = [(sim_table[p]) for p in u_neigh]  
            node_val = node_func(u_sims)
            edge_val = edge_func(graph.edges[(opt,u)]['weights'])
            neighbors_sim[u] = ((node_val * node_weight
                                + edge_val * edge_weight) 
                                / (node_weight + edge_weight))
        return neighbors_sim


    def traversal_cost(self,
                       initial_opt,
                       length,
                       edge_function=min,
                       weigth_function=min, 
                       remove_visited_edge=True):
        """
        This traversal algorithm selects the next vertex following the lowest
        cost edge.
        
            Generate a optimization sequence walking through the net
            Parameters:
            -----------
            initial_opt: str
              the optimization that will initialize the sequence
            length: int
              the aimed length of optmizaiton sequence
            edge_function: function(list) -> statistic of the list (e.g. len, min, max, etc)
                function used to get edge cost
            weigth_function: function(value1, value2) -> (value1 | value2) 
                function to decide the next node
                it is used to compare the edge cost of each adjacent node
            remove_visited_edge: boolean 
                if is true, the visited edge will be removed, otherwise
                there are no edges remove
        """
        g = copy.deepcopy(self.__Graph_Model.graph)
        seq = [initial_opt]
        opt = initial_opt
        while (len(seq) < length or 
               'terminal' not in g.nodes[opt]['attr']):

            if len(g.adj[opt]) == 0:
                break

            next_node=list(g.adj[opt])[0]
            next_val=edge_function(g.edges[(opt,next_node)]['weights'])
            
            for u in g.adj[opt]:
                u_val = edge_function(g.edges[(opt,u)]['weights'])
                if weigth_function(u_val,next_val) == u_val:
                    next_node = u
                    next_val = u_val
            seq.append(next_node)
            if remove_visited_edge:
                g.remove_edge(opt,next_node)
            opt = next_node
        
        return {'sequence':seq}
    
    def traversal_similar(self, 
                          initial_opt, 
                          length, 
                          sim_table,
                          sim_function=min,
                          compare_sim_function=min,
                          remove_visited_edge=True):
        """
        This traversal algorithm selects the next vertex based on
        the similarity between the training and testprograms. The next
        vertex, V.succ, is that labeled withthe optimization from Si,
        which belongs to the mostsimilar training program Pi.
        
        Parameters
        ----------
        initial_opt: str
              the optimization that will initialize the sequence
            length: int
              the aimed length of optmizaiton sequence
            sim_table: dict
                table of similarity between the test program and model program
                each dict key is a name of the train programs.
                each dict value is the similarity between the train program and
                test program.
            sim_function: function(*args) -> statistic of the args (e.g. min, max, mean, etc)
                used to decide the value that will "represent" the node similarity
                aplying this function into similarities of all programs stored into
                the node.
            compare_sim_function: function(value1, value2) -> (value1 | value2)
                used to compare two nodes deciding the next one in the path
            remove_visited_edge: boolean 
                if is true, the visited edge will be removed, otherwise
                there are no edges remove
        """
        seq = [initial_opt]
        opt = initial_opt
        g = copy.deepcopy(self.__Graph_Model.graph)
        while (len(seq) < length or 
               'terminal' not in g.nodes[opt]['attr']):

            if len(g.adj[opt]) == 0:
                break
            neighbors_sim = {}
            for u in g.adj[opt]:
                u_neigh = g.nodes[u]['programs']
                u_sims = [sim_table[p] for p in u_neigh] 
                neighbors_sim[u]=sim_function(u_sims)
            
            next_node = list(neighbors_sim)[0]
            for n in neighbors_sim:
                next_val = compare_sim_function(neighbors_sim[n],
                                                neighbors_sim[next_node])
                if next_val == neighbors_sim[n]:
                    next_node = n
            seq.append(next_node)
            if remove_visited_edge:
                g.remove_edge(opt,next_node)
            opt=next_node
        return {'sequence':seq}


    def traversal_weighted(self,
                           initial_opt,
                           length,
                           sim_table,
                           edge_weight=1.0,
                           node_weight=1.0,
                           edge_func=min,
                           node_func=min,
                           compare_sim_function=min,
                           remove_edge=True):
        """
        This traversal algorithm is based on the twoprevious algorithms.
        In this traversal, both strategies are applied (similar and cost) and
        the next vertex is chosen by the strategythat provides the lowest value
        between similarity andedge cost.

            enerate a optimization sequence walking through the net
            Parameters:

            initial_opt: str
              the optimization that will initialize the sequence
            length: int
              the aimed length of optmizaiton sequence
            sim_table: dict
                table of similarity between the test program and model program
                each dict key is a name of the train programs.
                each dict value is the similarity between the train program and
                test program.
            edge_weight: float
                the weight used to the edge into the 
            node_weight: float
            edge_func: function(lst) -> statistic of the list
                function to get the edge value (recieve a list of value
                and returns a statisc about it (min, max, size, etc))
            node_func:
                function to get the node value
            compare_sim_function: function(value1, value2) -> (value1 | value2)
                used to compare two nodes deciding the next one in the path 

        """
        g=copy.deepcopy(self.__Graph_Model.graph)
        seq = [initial_opt]
        opt=initial_opt
        while (len(seq) < length or 
              'terminal' not in g.nodes[opt]['attr']):

            if len(g.adj[opt]) == 0:
                break

            neighbors_sim = self.__neightbors_similarity(g, 
                                                        opt,
                                                        sim_table,
                                                        edge_weight,
                                                        node_weight,
                                                        node_func,
                                                        edge_func)
            next_node = list(neighbors_sim)[0]
            for n in neighbors_sim:
                next_val = compare_sim_function(neighbors_sim[n],
                                                neighbors_sim[next_node])
                if next_val == neighbors_sim[n]:
                    next_node = n
            seq.append(next_node)
            if remove_edge:
                g.remove_edge(opt,next_node)  
            opt=next_node
        return {'sequence':seq}

    def traversal_backtracking(self,
                               initial_opt, 
                               length, 
                               sim_table,
                               bench_dir, 
                               goals, 
                               get_compile_time=True,
                               edge_weight=1.0, 
                               node_weight=1.0, 
                               edge_func=min, 
                               node_func=min, 
                               sim_func=min):
        """
        This traversal algorithm evaluates the performance of the sub-sequences
        during the traversal, i.e. when the algorithm selects a next vertex it
        evaluates the new sub-sequence. This task aims to guide the traversal.
        Parameters:
            -----------
            initial_opt: str
              the optimization that will initialize the sequence
            length: int
              the aimed size of optmizaiton sequence
            sim_table: dict
                table of similarity between the test program and model program
                each dict key is a name of the train programs.
                each dict value is the similarity between the train program and
                test program.
            bench_dir: str
                directory of test benchmark
            goals: dict
                {goal: weight}
            get_compile_time: bool
                if true, store the compilation time and returns it on 
                as 'compilation_time' value 
        """
        def recursive_backtrack(g,
                                opt,
                                last_goal, 
                                seq, 
                                ignore_loss=False):
            str_seq = Sequence.name_pass_to_string(seq)
            nonlocal nc
            nonlocal sum_compile_time
            nonlocal clean
            nc += 1
            goal = Engine.evaluate(goals,
                                   str_seq,
                                   'opt',
                                   bench_dir,
                                   cleanup=clean)
            if get_compile_time:
                yl_name = os.path.join(bench_dir,
                                       'compile_time.yaml')
                compile_time_dict = IO.load_yaml(yl_name)
                sum_compile_time += compile_time_dict['compile_time']
                Engine.cleanup(bench_dir,
                               'opt')

            if not ignore_loss:
                if last_goal <= goal:
                    return []

            if (len(seq) >= length and 
                'terminal' in g.nodes[opt]['attr']):
                return seq
            elif len(g.adj[opt]) == 0:
                return seq
            else:
                neightbors_sim = self.__neightbors_similarity(g, 
                                                              opt,
                                                              sim_table,
                                                              edge_weight,
                                                              node_weight,
                                                              node_func,
                                                              edge_func)

                sorted_keys = sorted(neightbors_sim,
                                     key=neightbors_sim.__getitem__)
                find = False
                for optimization in sorted_keys:
                    new_seq=copy.deepcopy(seq)
                    new_seq.append(optimization)
                    ret = recursive_backtrack(g,
                                              optimization,
                                              goal,
                                              new_seq)
                    if ret != []:
                        find = True
                        break
                if find:
                    return ret
                else:
                    if 'router' in g.nodes[opt]['attr']:
                        optimization = sorted_keys[0]
                        new_seq=copy.deepcopy(seq)
                        new_seq.append(optimization)
                        return recursive_backtrack(g, 
                                                   optimization,
                                                   goal,
                                                   new_seq,
                                                   ignore_loss=True)
                    else:
                        return []

            
        nc = 0
        g = copy.deepcopy(self.__Graph_Model.graph)
        seq = [initial_opt]
        str_seq = Sequence.name_pass_to_string(seq)
        clean = not get_compile_time
        initial_goal = Engine.evaluate(goals,
                                        '-O0',
                                         'opt',
                                          bench_dir,
                                          cleanup=clean)
        if get_compile_time:
            yl_name = os.path.join(bench_dir,
                                   'compile_time.yaml')
            compile_time_dict = IO.load_yaml(yl_name)
            sum_compile_time = compile_time_dict['compile_time']
            Engine.cleanup(bench_dir,
                           'opt')
        else:
            sum_compile_time = None
        s = recursive_backtrack(g,
                                initial_opt,
                                float('inf'),
                                seq)
        
        ret = {'sequence': s,
               'compilation_count': nc}
        if get_compile_time:
            ret['compilation_time'] = sum_compile_time
        return ret