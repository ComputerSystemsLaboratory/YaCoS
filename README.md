# Welcome to YaCoS

The growing popularity of machine learning frameworks and algorithms has greatly contributed to the design and exploration of good code optimization sequences. Yet, in spite of this progress, mainstream compilers still provide users with only a handful of fixed optimization sequences. Finding optimization sequences that are good in general is challenging because the universe of possible sequences is potentially infinite. This paper describes a infrastructure that provides developers with the means to explore this space. Said infrastructure, henceforth called YaCoS, consists of benchmarks, search algorithms, metrics to estimate the distance between programs, and compilation strategies. YaCoS's features let users build learning models that predict, for unknown programs, optimization sequences that are likely to yield good results for them.

## Installation

YaCoS is compatible with Python 3.8, and is tested on Ubuntu 20.04. Other Linux distros should work as well, but Windows is not supported.

To install YaCoS, follow these two steps:


```
1. git clone https://github.com/ComputerSystemsLab/YaCoS.git
```

```
2. cd YaCoS
   ./install_deps.sh
   ./setup.py [test|build|install]
```

## Building Blocks

The building blocks provide search algorithms, as well as, functionalities for manipulating programs and objectives.

**1. Algorithms**

   - Benchmark reduction
   - Create sequences
     - Genetic Algorithm
     - Particle Swarm Optimization
     - Random
   - Find the best sequences
   - Sequence manipulation
     - Batch Elimination
     - Itertive Elimination
     - Combined Eliminatino
     - Improved Batch Elimination
     - Reduction

**2. Data Processing**

   - Clustering

**3. Essentials**

   - Benchmark manipulation (compile, execute, extract features, ...)
   - Objective/goal manipulation (single goal, multiple goals, ...)
   - IO manipulation (load, dump, open, ...)
   - Sequence manipulation (create, modification, ...)
   - Dataset manipulation (download)
   - Similarity (1D Euclidean distance, 2D Euclidean distance, edit distance, ...)

**4. Program Representation**

   - Sequences
     - Syntax sequence
     - Syntax tokens
     - Syntax tokens and variables
     - LLVM instructions sequence
     - Milepost Static features  
     - IR2Vec
   - Graphs
     - Abstract syntax tree
     - Abstract syntax tree + Data
     - Abstract syntax tree + Data + CFG Nodes
     - Control Flow Graph
     - Control Flow Graph + Call
     - Control Data Flow Graph
     - Control Data Flow Graph + Call
     - Control Data Flow Graph + Call + Basic Block Nodes
     - ProGraML
   - Others
     - Inst2vec

Such representations come from several researches:
- [Exploring the Space of Optimization Sequences for Code-Size Reduction: Insights and Tools.](https://doi.org/10.1145/3446804.3446849) Anderson Faustino da Silva, Bernardo N. B. de Lima and Fernando Magno Quintão Pereira. Compiler Construction. ACM SIGPLAN, 2021.
- [ComPy-Learn: A Toolbox for Exploring Machine Learning Representations for Compilers.](https://doi.org/10.1109/FDL50818.2020.9232946) Alexander Brauckmann, Andrés Goens and Jeronimo Castrillon. Forum for Specification and Design Languages. IEEE, 2020.
- [Neural Code Comprehension: A Learnable Representation of Code Semantics.](https://arxiv.org/abs/1806.07336) Tal Ben-Nun, Alice Shoshana Jakobovits and Torsten Hoefler. Advances in Neural Information Processing Systems 31. Curran Associates, 2018.
- [IR2Vec: LLVM IR Based Scalable Program Embeddings.](https://doi.org/10.1145/3418463) S. VenkataKeerthy, Rohit Aggarwal, Shalini Jain, Maunendra Sankar Desarkar, Ramakrishna Upadrasta and Y. N. Srikant. ACM Transactions on Architecture and Code Optimization, volume 17, number 4, 2020.

## Examples

 How to use YaCoS.

- **Extracting representation**
   - *AST graphs*
      - `./graph_from_ast.py <flags>`
      - `./graphs_from_ast.py <flags>`
   - *LLVM IR graphs*
      - `./graph_from_ir.py <flags>`
      - `./graphs_from_ir.py <flags>`
   - *Int2Vec*
      - `./inst2vec.py <flags>`
   - *IR2Vec*
     - `./ir2vec_from_ir.py <flags>`      
   - *Milepost Static Features*
      - `./milepost_from_ir.py <flags>`
   - *Sequence*
      - `./sequence.py <flags>`
- **Generating training data**
   - *Batch elimination*
      - `./batch_elimination.py <flags>`
   - *Iterative elimination*
      - `./batch_elimination.py <flags>`
   - *Combined elimination*
      - `./batch_elimination.py <flags>`
   - *Improved Batch elimination*
      - `./batch_elimination.py <flags>`
   - *Reduce benchmarks*
     - `./benchmark_reduction.py <flags>`
   - *Find the best k sequences from trained data*
     - `./best_k.py <flags>`
   - *Evaluate Partial sequences*
     - `./evaluate_partial_sequences.py <flags>`
   - *Evaluate k sequences*
     - `./evaluate_sequences.py <flags>`
   - *Extract the goal value for the compiler optimization levels*
     - `./levels.py <flags>`
   - *Generate and evaluate particle swarm optimization sequences*
     - `./pso.py`
   - *Generate and evaluate random sequences*
     - `./random_.py <flags>`
   - *Reduce sequences*
     - `./sequence_reduction.py <flags>`
   - *Generate and evaluate genetic sequences*
     - `./sga.py <flags>`
- **Implementing predictive compilation**
   - *Case-based reasoning strategy to find good sequences for a unseeen program*
     - `./cbr.py <flags>`
   - *Case-based reasoning strategy to find good sequences for a unseeen program (functions)*
     - `./cbr_function.py <flags>`
- **Tasks**
  - *Classify Applications*
    - `classifapp.py <flags>`
- **Others**
   - *Measure the distance between programs (Milepost representation)*
     - `./distance.py <flags>`
   - *Extract the cost of each function*
     - `./function_cost.py <flags>`
   - *Extract the name of the variables and functions*
     - `./globals.py <flags>`

## Benchmarks

The user can download pre-configured benchmarks.

- <a href="http://cuda.dcc.ufmg.br/angha/about">AnghaBench</a><a href="http://www.csl.uem.br/repository/data/AnghaBench.tar.xz"><img src="images/download.png" alt="Donwnload" width=32 height=32></a>
- <a href="http://vhosts.eecs.umich.edu/mibench/">MiBench</a><a href="http://www.csl.uem.br/repository/data/MiBench.tar.xz"><img src="images/download.png" alt="Donwnload" width=32 height=32></a>
- <a href="https://www.embench.org/">Embench</a><a href="http://www.csl.uem.br/repository/data/embench-iot.tar.xz"><img src="images/download.png" alt="Donwnload" width=32 height=32></a>

## Publication

André Felipe Zanella, Anderson Faustino da Silva and Fernando Magno Quintão. 2020. [YACOS: a Complete Infrastructure to the Design and Exploration of Code Optimization Sequences.](https://doi.org/10.1145/3427081.3427089) Proceedings of the 24th Brazilian Symposium on Programming Languages. Association for Computing Machinery, New York, NY, USA, 56–63.

## Contact

_Anderson Faustino da Silva_ (csl@uem.br)
