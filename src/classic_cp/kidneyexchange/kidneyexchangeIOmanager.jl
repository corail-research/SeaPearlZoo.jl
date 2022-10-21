struct KidneyExchangeInputData
    numberOfPairs         :: Int
    density               :: Float16
    compatibilities       :: Vector{Vector{Int}}
end

"""
parseInput(raw_input)

Return an KidneyExchangeInputData with the following fields:
- numberOfPairs::Int                 -> number of donor-patient pairs
- density::Float16                   -> probability of compatibility between 2 pairs (density ∈ [0,1])
- compabilities::Vector{Vector{Int}} -> v[i] contains the pairs that can give a kidney to pair i  

# Arguments
- `filename`: file containing the compatibilities between pairs

# Instance example: 
4 0.33 #numberOfPairs density
2 4    #pair 1 can receive a kidney from pairs 2 and 4
1      #pair 2 can receive a kidney from pair 1
4      #pair 3 can receive a kidney from pair 4
3      #pair 4 can receive a kidney from pair 3
"""
function parseKidneyExchangeInput(raw_input)
    lines = split(raw_input, '\n')
    firstLine = split(lines[1], ' ')

    numberOfPairs = parse(Int, firstLine[1])
    density = parse(Float16, firstLine[2])

    compabilities = Vector{Vector{Int}}(undef, numberOfPairs)

    #TODO remove hypothetical white space at the end of lines
    for i = 2:numberOfPairs+1
        if isempty(lines[i])
            #Edge case: pair i can't receive any kidney
            compabilities[i-1] = []
        else
            line = split(lines[i],' ')
            compabilities[i-1] = map(x->parse(Int,x), line)
        end
    end
    return KidneyExchangeInputData(numberOfPairs, density, compabilities)
end

"""
getInputData(filename::String)

return the KEP instance corresponding to filemane

# Arguments
- `filename`: file containing the compatibilities between pairs
"""
function getKidneyExchangeInputData(filename::String)
    inputData = nothing
    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end

    open(filename, "r") do openedFile
        inputData = parseKidneyExchangeInput(read(openedFile, String))
    end
    return inputData
end

"""
print_solutions_matrix(solved_model::SeaPearl.CPModel; max_nb_sols=nothing)

By default, print all solutions (in matrix form and as a set of cycles) calculated by solve_kidneyexchange_matrix()
If `max_nb_sols` is setted, we print only `max_nb_sols` best solutions
If `isReduced` is true, `print_pairsEquivalence()` is called

# Print format (for each solution)
- matrix form
    m[i, j] = 1 => pair i receive a kidney from pair j
    m[i, j] = 0 => pair i do not receive a kidney from pair j
- set of cycles form
    4 -> 7 -> 1 => pair 4 gives a kidney to pair 7, pair 7 gives a kidney to pair 1 and pair 1 gives a kidney to pair 4
"""
function print_solutions_matrix(solved_model::SeaPearl.CPModel; max_nb_sols=nothing)
    #Filter solutions to remove "nothing" 
    solutions = solved_model.statistics.solutions
    if isdefined(solved_model, :adhocInfo)
        isReduced = true
        pairsEquivalence = solved_model.adhocInfo
    else
        isReduced = false
        pairsEquivalence = nothing
    end
    numberOfPairs = trunc(Int, sqrt(length(solved_model.variables) - 1))
    realSolutions = filter(e -> !isnothing(e),solutions)

    if isnothing(max_nb_sols)
        # Print all solutions
        solutionsToPrint = realSolutions
    else
        # Print `nb_sols` best solutions
        @assert max_nb_sols ≥ 1 
        nbSolutions = length(realSolutions)
        solutionsToPrint = realSolutions[max(1, nbSolutions - max_nb_sols + 1):nbSolutions]
    end
 
    #Print solutions
    counter = 1
    objective = solved_model.objective.id
    for solution in solutionsToPrint
        score = solution[objective]
        println("### nº"*string(counter)*" -> "*string(score * -1)*" exchanges  ###")
        # Print matrix
        print("Solution as a matrix")
        counter += 1
        if isReduced print(" (reduced instance)") end
        println()
        coordOnes = []
        for i in 1:numberOfPairs
            for j in 1:numberOfPairs
                val = string(solution["x_"*string(i)*"_"*string(j)])
                print(val*" ")
                if val == "1"
                    push!(coordOnes, (i, j))
                end
            end
            println()
        end
        println()

        #Print cycles
        print_cycles(coordOnes, isReduced, pairsEquivalence)
   
    end

    #Print pairsEquivalence (link between the original pairs and the reduced pairs)
    if isReduced
        if isReduced
            print_pairsEquivalence(pairsEquivalence)
        end
    end
    
end

"""
print_solutions_vector(solved_model::SeaPearl.CPModel; max_nb_sols=nothing)
By default, print all solutions (in vector form and as a set of cycles) calculated by solve_kidneyexchange_vector()
If `max_nb_sols` is setted, we print only `max_nb_sols` best solutions
If `isReduced` is true, `print_pairsEquivalence()` is called
# Print format (for each solution)
- vector form
    v[i] = j => pair j receive a kidney from pair i
- set of cycles form
    4 -> 7 -> 1 => pair 4 gives a kidney to pair 7, pair 7 gives a kidney to pair 1 and pair 1 gives a kidney to pair 4
"""
function print_solutions_vector(solved_model::SeaPearl.CPModel; max_nb_sols=nothing)
    #Filter solutions to remove "nothing" 
    solutions = solved_model.statistics.solutions
    if isdefined(solved_model, :adhocInfo)
        isReduced = true
        pairsEquivalence = solved_model.adhocInfo
    else
        isReduced = false
        pairsEquivalence = nothing
    end
    numberOfPairs = trunc(Int, length(solved_model.variables) - 1)
    realSolutions = filter(e -> !isnothing(e),solutions)

    if isnothing(max_nb_sols)
        # Print all solutions
        solutionsToPrint = realSolutions
    else
        # Print `nb_sols` best solutions
        @assert max_nb_sols ≥ 1 
        nbSolutions = length(realSolutions)
        solutionsToPrint = realSolutions[max(1, nbSolutions - max_nb_sols + 1):nbSolutions]
    end

    counter = 1
    objective = solved_model.objective.id
    for solution in solutionsToPrint
        score = solution[objective]
        println("### nº"*string(counter)*" -> "*string(score * -1)*" exchanges  ###")
        counter += 1

        #Print vector
        print("Solution as a vector")
        if isReduced print(" (reduced instance)") end
        vector_solution = [solution["x_"*string(i)] for i in 1:numberOfPairs]
        println()
        println(vector_solution)
        println()

        #Print cycles
        tupleSolution = [(i, vector_solution[i]) for i in 1:numberOfPairs]
        print_cycles(tupleSolution, isReduced, pairsEquivalence)
    end

    #Print pairsEquivalence (link between the original pairs and the reduced pairs)
    if isReduced
        print_pairsEquivalence(pairsEquivalence)
    end
end

"""
print_cycles(exchangeTuples)

Print cycles that make up the optimal solution 

# Example
If the solution is composed by one cycle as "pair 4 gives a kidney to pair 7, pair 7 gives a kidney to pair 1 and pair 1 gives a kidney to pair 4", the function will print:
Cycle of size 3: 4 -> 7 -> 1 
"""
function print_cycles(exchangeTuples, isReduced, pairsEquivalence)
    print("Solution as a set of cycles")
    if isReduced print(" (original instance)") end
    println()
    cycles = []
    
    #Find cycles
    while !isempty(exchangeTuples)
        current = pop!(exchangeTuples)
        cycle = []
        push!(cycle, current)
        if current[1] != current[2]
            isOpen = true
        else
            #Edge case: pair i is compatible with itself
            isOpen = false
        end

        while isOpen
            idx = findfirst(x -> x[1] == cycle[end][2], exchangeTuples)
            current = splice!(exchangeTuples, idx)
            push!(cycle, current)
            if cycle[1][1] == cycle[end][2]
                isOpen = false
            end
        end
        push!(cycles, cycle)
    end

    #Print cycles
    for cycle in cycles
        if length(cycle) > 1
            print("Cycle of size "*string(length(cycle))*": ")
            for pair in cycle
                isReduced ? print(string(pairsEquivalence[pair[1]])) : print(string(pair[1]))
                if pair != cycle[end]
                    print(" -> ")
                end
            end
            println()
        end
    end
    println()
end

"""
print_pairsEquivalence(pairsEquivalence)

Print pairsEquivalence (link between the original pairs and the reduced pairs)

# Example
If the original instance has 9 pairs and that pairs 1, 4, 6, 7 and 8 have been removed, the function will print:
Table of equivalence: Original pair | Reduced pair
2  | 1
3  | 2
5  | 3
9  | 4
"""
function print_pairsEquivalence(pairsEquivalence)
    println("### Table of equivalence ###")
    println("Original pair | Reduced pair")
        for i in 1 : length(pairsEquivalence)
            #Pad with whitespace to align values
            firstPair = string(pairsEquivalence[i])
            formatedFirstPair = firstPair*" "^(length(string(pairsEquivalence[end]))-length(firstPair))
            println(formatedFirstPair*" | "*string(i))
        end
        println()
        println()
end