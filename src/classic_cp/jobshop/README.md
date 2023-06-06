# Job-shop scheduling (JS)

This is inspired from [the personal MiniZinc page of Hakan Kjellerstrand](http://www.hakank.org/minizinc/).
One can find information about the Job-shop scheduling [here] (https://en.wikipedia.org/wiki/Job-shop_scheduling)
One can find solved instances for the Job-shop scheduling [here] (https://github.com/tamy0612/JSPLIB/tree/master/instances)

In this version of JS, we consider:
<ul>
  <li>No time interval between two tasks is necessary</li>
  <li>Each job has to go through each machine in an order given by the instance</li>
  <li>Deterministic (fixed) processing times</li>
  <li>The objective function is to minimize the makespan (i.e. the biggest end time)</li>
</ul>

## Installation

To launch this example, you need to have the package `SeaPearl` added to your environment.

## Usage

Being inside that folder in the terminal (`classic_cp/jobshop/`), you can launch:

```julia
julia> include("jobshop.jl");
julia> solved_model = solve_jobshop("data/js_3_3");
julia> print_solution(solved_model);
```

This will solve the problem "js_3_3" and print the solution found to the terminal.

## Instance example: 

# numberOfMachines, numberOfJobs, maxTime
3 3 500     

# duration of each task (i.e. 54 is the duration of task 1 on machine 2)
23 54 19    
45 74 34
63 73 55

# order of each job (i.e. job 1 has to follow this order: machine 3, machine 2, machine 1)
2 1 0       
0 2 1
1 2 0