# n queens problem

The nqueens problem is the generalization of the well known 8 queens problem on a n\*n chess board. The eight queens puzzle is the problem of placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other; thus, a solution requires that no two queens share the same row, column, or diagonal.


This part uses RL agent to determine a good graph representation of the nqueen problem using GNN and learn a good value selection heuristic for a given size n of the nqueens problem. The learned heuristic is obviously over-specialized on the given instance size.   One can define its own graph modelling, RL agent or reward to drive the research and the learning. 

This code is given as an exemple of what kind of value selection heuristic can be approximate / learned using Deep Reinforcement Learning Agent but doesn't pretend to be able to generalize over problem of different size.    


## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`examples/nqueens/`), you can launch:

```julia
julia> include("nqueens.jl")
julia> #TODO complete readME.md
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
