# Kidney Exchange Problem (KEP)

This is inspired from [the personal MiniZinc page of Hakan Kjellerstrand](http://www.hakank.org/minizinc/).

In this version of KEP, we consider:
<ul>
  <li>Only cycles (no chains)</li>
  <li>Unweighted arcs</li>
  <li>No upper bound for the size of the cycles</li>

The objective is to maximize the number of exchanges.

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`classic_cp/kidneyexchange/`), you can launch:

```julia
julia> include("kidneyexchange.jl");
julia> model_solved = solve_kidneyexchange("data/kep_8_0.2");
julia> print_solutions(model_solved);
```

This will print the solutions found as a matrix and a list of cycles.

## Instance example: 
4 0.33 #numberOfPairs density
2 4    #pair 1 can receive a kidney from pairs 2 and 4
1      #pair 2 can receive a kidney from pair 1
4      #pair 3 can receive a kidney from pair 4
3      #pair 4 can receive a kidney from pair 3