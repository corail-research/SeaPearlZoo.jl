# n queens problem

The nqueens problem is the generalization of the well known 8 queens problem on a n\*n chess board. The eight queens puzzle is the problem of placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other; thus, a solution requires that no two queens share the same row, column, or diagonal.


This version is a classic version of the nqueens problem ( meaning that heuristics used are fully deterministic ). One can use predefined different heuristic for variable or value selection, or can even define its own variable or value selection heuristics.

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`classic_cp/nqueens/`), you can launch:

```julia
julia> include("nqueens.jl");
julia> model_solved = solve_queens(8);
julia> print_queens(model_solved;nb_sols=5);
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
