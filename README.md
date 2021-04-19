# SeaPearlZoo

This project provides some examples of SeaPearl.jl's utilisation. One will find good practices to run experiences with SeaPearl.jl.

## Launch graph coloring learning 

First, clone SeaPearlZoo locally:
```bash
$ git clone https://github.com/corail-research/SeaPearlZoo
```

Now, making sure you have Julia 1.6 installed, go to the [Graph-Coloring](https://en.wikipedia.org/wiki/Graph_coloring) directory.
```bash
$ cd SeaPearlZoo/learning_cp/graphcoloring
```

Now, activate and instantiate (download and install all the packages) the environment:
```
$ julia
julia> ]
(@v1.6) pkg> activate . 
(graphcoloring) pkg> instantiate
```

Go back to the [julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) by pressing backspace.

Finally, you can launch the `graphcoloring.jl` file :
```julia
julia> include("graphcoloring.jl")
```

Here you should see a ProgressBar showing you the evolution of the training and then the progression of the benchmarking. You will end up with csv files stored in the same directory to measure the performances. The trained weights will also be stored in a BSON file.

## Repo organisation

- classic_cp (just constraint programming problems without learning)
    - graphcoloring
    - knapsack
- learning_cp (using SeaPearl.jl to have an agent learn a value selection heuristic)
    - graphcoloring
    - knapsack


## Learning example organisation

- graphcoloring/
    - agents.jl
    - features.jl
    - graphcoloring.jl (main file)
    - rewards.jl
    - Project.toml (here lies the dependencies of the example, used when you activate the environment)

