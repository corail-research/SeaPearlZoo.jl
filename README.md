# SeaPearlZoo

This project provides some examples of SeaPearl.jl's utilisation. One will find good practices to run experiences with SeaPearl.jl. 

## Launch graph coloring learning 
First, clone SeaPearl.jl locally:
```bash
$ git clone git@github.com:ilancoulon/CPRL.jl.git
```

Then, clone SeaPearlZoo locally:
```bash
$ git clone git@github.com:ilancoulon/SeaPearlZoo.git
```

Now, make sure julialang is [installed](https://julialang.org/downloads/) on your computer and go to learning graph coloring example. (if you don't know the graph coloring problem, here is the wikipedia [link](https://en.wikipedia.org/wiki/Graph_coloring)).
```bash
$ cd SeaPearlZoo/learning_cp/graphcoloring
```

Now, make SeaPearl.jl a dependence of the project: 
(launch julia, go to the [Pkg REPL](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html), activate the environment and create the dependency)
```
$ julia
julia> ]
(@v1.4) pkg> activate . 
(graphcoloring) pkg> dev path/to/SeaPearl.jl
```

Go back to the [julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) by pressing backspace.

Finally, launch the graphcoloring.jl file :
```julia
julia> include("graphcoloring.jl")
```

Here you should see a ProgressBar showing you the evolution of the training and then the progression of the benchmarking and at the end a new window should open with plots describing what happened.

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

