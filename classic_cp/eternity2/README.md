# Eternity2 Problem

The Eternity 2 puzzle is an edge matching problem. It involves placing square puzzle pieces constrained by the requirement to match adjacent edges. https://en.wikipedia.org/wiki/Eternity_II_puzzle


This version is a classic version of the eternity problem (meaning that heuristics used are fully deterministic). One can use predefined different heuristic for variable or value selection, or can even define its own variable or value selection heuristics.

Here, we developed three different kinds of modeling to compare the performances.

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`examples/eternity2/`), you can launch:

```julia
julia> include("eternity2.jl.jl")
julia> solve_coloring("data/eternity3x3")
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
