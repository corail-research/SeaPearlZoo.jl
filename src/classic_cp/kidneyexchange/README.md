# Kidney Exchange Problem (KEP)

This is inspired from [the personal MiniZinc page of Hakan Kjellerstrand](http://www.hakank.org/minizinc/).

In this version of KEP, we consider:
<ul>
  <li>Only cycles (no chains)</li>
  <li>Unweighted arcs</li>
  <li>No upper bound for the size of the cycles</li>
</ul>
The objective is to maximize the number of exchanges.

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`classic_cp/kidneyexchange/`), you can launch:

### Model A (solution as a matrix)

This model does allow self-compatible pairs. If donor from pair i gives a kidney to patient of pair i, it will be a size 1 cycle represented as a 1 in the i-th element of the diagonal of the matrix solution.

```julia
julia> include("kidneyexchange.jl");
julia> model_solved = solve_kidneyexchange_matrix("data/kep_8_0.2");
julia> print_solutions_matrix(model_solved);
```

This will print the solutions found as a matrix and a list of cycles.

### Model B (solution as a vector)

This model does not allow self-compatible pairs, as there will be no way to distinguish a self-compatible pair (size one cycle) from a pair that does not participate in any cycle. 

```julia
julia> include("kidneyexchange.jl");
julia> model_solved = solve_kidneyexchange_vector("data/kep_8_0.2");
julia> print_solutions_vector(model_solved);
```

This will print the solutions found as a vector and a list of cycles.

## Instance example: 
```
4 0.33 #numberOfPairs density
2 4    #pair 1 can receive a kidney from pairs 2 and 4
1      #pair 2 can receive a kidney from pair 1
4      #pair 3 can receive a kidney from pair 4
3      #pair 4 can receive a kidney from pair 3
```
## Output example: 
```
julia> model_solved = solve_kidneyexchange_vector("data/kep_13_0.1");
julia> print_solutions_vector(model_solved; max_nb_sols=2);
### nº1 -> 3 exchanges  ###
Solution as a vector (reduced instance)
[3, 1, 2, 4, 5, 6]

Solution as a set of cycles (original instance)
Cycle of size 3: 4 -> 3 -> 2

### nº2 -> 6 exchanges  ###
Solution as a vector (reduced instance)
[3, 1, 2, 6, 4, 5]

Solution as a set of cycles (original instance)
Cycle of size 3: 13 -> 12 -> 11
Cycle of size 3: 4 -> 3 -> 2

### Table of equivalence ###
Original pair | Reduced pair
2  | 1
3  | 2
4  | 3
11 | 4
12 | 5
13 | 6
```
