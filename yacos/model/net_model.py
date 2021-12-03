"""
Copyright 2021 Nilton Luiz Queiroz Junior and Anderson Faustino da Silva.

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



import networkx
import os
import copy
import pickle
import yaml

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence
from yacos.essential import Similarity

class NetModel:
    """ Create the Net Model """
    def __init__(self):
        """
        Initialize a Net Model objetct.
        """

        self._graph = networkx.DiGraph()
        self._initial_nodes = {}
        self._representations = {}

    @property
    def graph(self):
        return self._graph
    
    @property
    def initial_nodes(self):
        return self._initial_nodes


    @property
    def representations(self):
        return self._representations
        

    def insert_sequences(self, program_list):
        """
            insert a list of optimization into NetModel
            Parameters
            ---------
            program_list:
            	a list of tuples 
		each tuple is composed by (benchmark_name, performance_dictionary, representation)
            		benchmark_name: str
            		performance_dictionary: dictionary
                	see it keys and values at add_sequence method
        """

        for (prog, perf_dict, rep) in program_list:
            self.add_sequence(perf_dict,prog,rep)

    def add_sequence(self,
                     performance_dict,
                     program_name,
                     rep):
        """
        Add a sequence into net

        Parameters
        ----------
        performance_dict: dict
            {iteration:{'seq': list
                        'goal': float}}
            where the list stored in seq is a list with optimizations
            containing a sequnce compiled with optimizations in position
            0 .. iteration+1. (iteration starts in 0)
            example:
            for the program X we have a performance_dict
            this performace dict can be:
            {0 : {'seq': ['first_opt']
                  'goal': float}
             1 : {'seq': ['first_opt','second_opt']
                  'goal': float}}
        program_name:str
            name of program associated with the sequence stored into performance dict
        rep : ?
            representation of the program associated with the sequence
            The representation doesn't have a defined type, it is flexible
        """

        
        self._representations[program_name] = rep
        sequence_size = len(performance_dict)

        #insert nodes on the net
        for idx, data in performance_dict.items():
            opt = data['seq'][idx]
            if idx == 0:
                attribute = 'initial'
            elif idx == sequence_size-1:
                attribute = 'terminal'
            else:
                attribute = 'complement'

            if opt in self._graph.nodes():
                self._graph.nodes[opt]['attr'].add(attribute)
            else:
                self._graph.add_node(opt,
                                     attr=set([attribute]),
                                     initial_of=set(),
                                     programs=set())
            
            if attribute == 'initial' and program_name != None:
                self._graph.nodes[opt]['initial_of'].add(program_name)
                self._initial_nodes[program_name] = opt
            self._graph.nodes[opt]['programs'].add(program_name)


        #build a dictionary {idx: {'goal': float, 'name':string}}
        # goal is the value of goal for optimization until position idx
        # name is the  optimization of sequence in position idx
        opt_goals = {}
        for idx in performance_dict.keys():
            opt_goal = performance_dict[idx]['goal']
            opt_name = performance_dict[idx]['seq'][-1]
            opt_goals[idx] = {}
            opt_goals[idx]['goal'] = opt_goal
            opt_goals[idx]['name'] = opt_name
        
        #insert edges on the graph and find routers nodes
        for i in list(opt_goals.keys())[:-1]:
            u_pair = opt_goals[i]
            v_pair = opt_goals[i+1]
            u = u_pair['name']
            v = v_pair['name']
            w = v_pair['goal']-u_pair['goal']
            if (u,v) in self._graph.edges():
                self._graph.edges[(u,v)]['weights'].append(w)
            else:
                self._graph.add_edge(u,
                                     v,
                                     weights=[w])
            if len(self._graph.adj[u]) > 1:
                self._graph.nodes[u]['attr'].add('router')

    def initial_opt_of(self,
                       program):
        """
            returns the initial optimization of the program
        """
        return self._initial_nodes[program]

    def save_net(self,
                 file_name,
                 repr_file=None):
        if float(networkx.__version__[:3]) <= 2.5:
            networkx.write_yaml(self._graph,
                                file_name)
        else:
            f=open(file_name,'w')
            yaml.dump(self._graph,f)
            f.close()

        if repr_file != None:
            out_file = open(repr_file,'wb')
            pickle.dump(self._representations,
                        out_file)
            out_file.close()

    def load_net(self,
                 file_name,
                 repr_file=None):

        if float(networkx.__version__[:3]) <= 2.5: 
            self._graph = networkx.read_yaml(file_name)
        else:
            f=open(file_name,'r')
            self._graph = yaml.load(f, Loader=yaml.Loader)
            f.close()

        for n in self._graph:
            for p in self._graph.nodes[n]['initial_of']:
              if p != None:
                self._initial_nodes[p] = n
        if repr_file != None:
            in_file = open(repr_file,'rb')
            self._representations = pickle.load(in_file)
            in_file.close()


    
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
    
    def average_weighted_walk(self,
                              initial_opt,
                              size,
                              sim_table,
                              edge_weight=1.0,
                              node_weight=1.0,
                              edge_func=min,
                              node_func=min,
                              sim_func=min,
                              remove_edge=False):
        """
            Generate a optimization sequence walking through the net
            Parameters:

            initial_opt: str
              the optimization that will initialize the sequence
            size: int
              the aimed size of optmizaiton sequence
            edge_weight: float
            node_weight: float
            edge_func: function to get the edge value (recieve a list of value and returns a statisc about it (min, max, size, etc) )
            node_func: function tp get the node value
        """
        g=copy.deepcopy(self._graph)
        seq = [initial_opt]
        opt=initial_opt
        while (len(seq) < size or 
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
                next_val = sim_func(neighbors_sim[n],
                                    neighbors_sim[next_node])
                if next_val == neighbors_sim[n]:
                    next_node = n
            seq.append(next_node)
            if remove_edge:
                g.remove_edge(opt,next_node)  
            opt=next_node
        return seq

    def cost_walk(self, 
                  initial_opt, 
                  size, 
                  edge_function=min,
                  weigth_function=min, 
                  remove_edge=False):
        """
            Generate a optimization sequence walking through the net
            Parameters:

            initial_opt: str
              the optimization that will initialize the sequence
            size: int
              the aimed size of optmizaiton sequence
            edge_function: function used to get edge value
            weigth_function: function to decide the next node
        """
        g = copy.deepcopy(self._graph)
        seq = [initial_opt]
        opt = initial_opt
        while (len(seq) < size or 
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
            if remove_edge:
                g.remove_edge(opt,next_node)
            opt = next_node
        
        return seq

    def similarity_walk(self, 
                        initial_opt, 
                        size, 
                        sim_table,
                        remove_edge=False, 
                        sim_function=min):
        """

          sim_function: recebe uma lista de similaridades e retorna uma delas

        """
        
        seq = [initial_opt]
        opt = initial_opt
        g = copy.deepcopy(self._graph)
        while (len(seq) < size or 
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
                next_val = sim_function(neighbors_sim[n],
                                        neighbors_sim[next_node])
                if next_val == neighbors_sim[n]:
                    next_node = n
            seq.append(next_node)
            if remove_edge:
                g.remove_edge(opt,next_node)
            opt=next_node
        return seq

    def improve_similarity_walk(self,
                                initial_opt, 
                                size, 
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
            walks trhough graph extracting the representation of programs 
            during the walk 
            This walks get objective on each router node and only goes on
             when it gets improvements

            Parameters:
            -----------
            initial_opt: str
              the optimization that will initialize the sequence
            size: int
              the aimed size of optmizaiton sequence
            goals: dict
                {goal: weight}
        """
        def recursive_improve_walk(g, opt, last_goal, seq, ignore_loss=False):
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
                compile_time_dict = IO.load_yaml(bench_dir+'/compile_time.yaml')
                sum_compile_time += compile_time_dict['compile_time']
                Engine.cleanup(bench_dir,
                               'opt')

            if not ignore_loss:
                if last_goal <= goal:
                    return []

            if (len(seq) >= size and 
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
                    ret = recursive_improve_walk(g,
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
                        #seq.append(optimization)
                        
                        new_seq=copy.deepcopy(seq)
                        new_seq.append(optimization)
                        return recursive_improve_walk(g, 
                                                      optimization,
                                                      goal,
                                                      new_seq,
                                                      ignore_loss=True)
                    else:
                        return []

            
        nc = 0
        g = copy.deepcopy(self._graph)
        seq = [initial_opt]
        str_seq = Sequence.name_pass_to_string(seq)
        clean = not get_compile_time
        initial_goal = Engine.evaluate(goals,
                                        '-O0',
                                         'opt',
                                          bench_dir,
                                          cleanup=clean)
        if get_compile_time:
            compile_time_dict = IO.load_yaml(bench_dir+'/compile_time.yaml')
            sum_compile_time = compile_time_dict['compile_time']
            Engine.cleanup(bench_dir,
                           'opt')
        print('-O0 goal:',initial_goal)
        s = recursive_improve_walk(g,
                                   initial_opt,
                                   float('inf'),
                                   seq)
        print('numero de compilacoes:',nc)
        if get_compile_time:
            print('tempo de compilacao:', sum_compile_time)
        return s

    


