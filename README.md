# SeaPearlZoo

This project provides some examples of [SeaPearl.jl](https://github.com/corail-research/SeaPearl.jl)'s utilisation. One will find good practices to run experiences with SeaPearl.jl.

## Launch graph coloring learning 

You need to have julia 1.8 installed.

First, clone SeaPearlZoo locally:
```bash
$ git clone https://github.com/corail-research/SeaPearlZoo.jl
```
If a Manifest.toml file is already present in your repo, delete it.

Now, activate and instantiate (download and install all the packages) the environment:
```
$ julia
julia> ]
(@v1.7) pkg> activate . 
(graph_coloring) pkg> instantiate
```

Go back to the [julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) by pressing backspace.

Finally, you can launch the `graph_coloring.jl` file :
```julia
julia> include("src/learning_cp/graph_coloring/graph_coloring.jl")
```

Here you should see a ProgressBar showing you the evolution of the training and then the progression of the benchmarking. You will end up with csv files stored in the same directory to measure the performances. The trained weights will also be stored in a BSON file.
```



## Repo organisation

- classic_cp (just constraint programming problems without learning)
    - All-interval_Series
    - graph_coloring
    - eternity2
    - jobshop
    - kidney_exchange
    - knapsack
    - latin
    - nqueens
    - tsptw
- learning_cp (using SeaPearl.jl to have an agent learn a value selection heuristic)
    - eternity2
    - graph_coloring
    - kidney_exchange
    - knapsack
    - latin
    - nqueens
    - tsptw
