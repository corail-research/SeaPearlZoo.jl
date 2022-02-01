# Kidney Exchange Problem (KEP)

This is inspired from [the personal MiniZinc page of Hakan Kjellerstrand](http://www.hakank.org/minizinc/).

In this version of KEP, we consider:
<ul>
  <li>only cycles (no chains)</li>
  <li>unweighted arcs</li>
  <li>no upper bound for the size of the cycles</li>

The objective is to maximize the number of exchanges.

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`classic_cp/kidneyexchange/`), you can launch:

```julia
julia> include("kidneyexchange.jl")
julia> model_solved = solve_kidneyexchange("data/kep_8_0.2")
julia> print_solutions(model_solved; nb_sols=1)
```

This will print the solutions found as a matrix and a list of cycles.