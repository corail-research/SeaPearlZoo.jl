using SeaPearl
struct InputData
    numberOfPairs         :: Int
    density               :: Float16
    compatibilities       :: Vector{Vector{Int}}
end

include("IOmanager.jl")

"""
solve_kidneyexchange_vector(filename::String)

Return the SeaPearl model solved (solution as a vector) for to the KEP problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic
This model does not allow self-compatible pairs, as there will be no way to distinguish a self-compatible pair (size one cycle) from a pair that does not participate in any cycle. 

# Arguments
- `filename`: file containing the compatibilities between pairs

# Constraints
- ReifiedInSet
- NotEqualConstant
- AllDifferent
- SumToZero

# Objective: maximize the number of valid exchanges

# Branchable variables
- n size vector where: 
    n is the reduced number of pairs
    v[i] = j => pair j receive a kidney from pair i
    v[i] = i => pair i does not participate in any cycle
"""
function solve_kidneyexchange_vector(filename::String)
    inputData = getInputData(filename)
    c = inputData.compatibilities
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Attempting to reduce the instance ###    
    canReduce, pairsEquivalence, reduced_c = reduce_instance(inputData)
    if canReduce
        model.adhocInfo = pairsEquivalence
        c = reduced_c
    end
    n = length(c)
    
    ### Variable declaration ###    

    # x[i] = j => pair j receive a kidney from pair i
    # x[i] = i => pair i ∉ cycles
    x = Vector{SeaPearl.AbstractIntVar}(undef, n) 

    # "fake" variable to know if a pair participate in a cycle
    index = Vector{SeaPearl.AbstractVar}(undef, n) 

    # "view" variable to be able to use the "ReifiedInSet" constraint
    # xEqualIndex_Bool[i] = true => i ∉ cycle
    # xEqualIndex_Bool[i] = false => i ∈ cycle
    xEqualIndex_Bool = Vector{SeaPearl.AbstractBoolVar}(undef, n)  

    # "view" variable to be able to use the "ReifiedInSet" constraint
    # xNotEqualIndex_Int[i] = 1 => i ∈ cycle
    # xNotEqualIndex_Int[i] = 0 => i ∉ cycle
    xNotEqualIndex_Int = Vector{SeaPearl.AbstractIntVar}(undef, n) 

    # "fake" variable to know if xNotEqualIndex_Int == 0
    zeroSet = SeaPearl.IntSetVar(0, 0, "zeroSet", model.trailer)
    # require the unique value to ensure that zeroSet is bounded
    SeaPearl.require!(zeroSet.domain, 0)

    for i = 1:n
        x[i] = SeaPearl.IntVar(1, n, "x_"*string(i), model.trailer)
        SeaPearl.addVariable!(model, x[i]; branchable=true)

        index[i] = SeaPearl.IntSetVar(i, i, "index_"*string(i), model.trailer)
        # require the unique value to ensure that isbound(index[i]) return true
        SeaPearl.require!(index[i].domain, i)
        
        xEqualIndex_Bool[i] = SeaPearl.BoolVar("xEqualIndex_Bool_"*string(i), model.trailer)
        xNotEqualIndex_Int[i] = SeaPearl.IntVar(0, 1,"xNotEqualIndex_Int_"*string(i), model.trailer)
    end

    ### Constraints ###
    for i = 1:n
        #Fix xEqualIndex_Bool
        SeaPearl.addConstraint!(model, SeaPearl.ReifiedInSet(x[i], index[i], xEqualIndex_Bool[i], trailer))

        #Fix xNotEqualIndex_Int
        SeaPearl.addConstraint!(model, SeaPearl.ReifiedInSet(xNotEqualIndex_Int[i], zeroSet, xEqualIndex_Bool[i], trailer))

        #Add incompatibilities to the model
        for j in 1:n
            if j ∉ c[i] && i !=j
                SeaPearl.addConstraint!(model, SeaPearl.NotEqualConstant(x[i], j, trailer))
            end
        end
    end

    #Check that any pair receives/gives more than 1 kidney and that for each pair "give a kidney <=> receive a kidney"
    SeaPearl.addConstraint!(model, SeaPearl.AllDifferent(x, trailer))

    ### Objective ###

    #SeaPearl's solver minimize the objective variable, so we use minusNumberOfExchanges in order to maximize the number of exchanges
    minusNumberOfExchanges = SeaPearl.IntVar(-n, 0, "minusNumberOfExchanges", trailer) 
    SeaPearl.addVariable!(model, minusNumberOfExchanges; branchable=false)
    vars = SeaPearl.AbstractIntVar[]

    #Concatenate all values of xNotEqualIndex_Int and minusNumberOfExchanges
    for i in 1:n
        vars = cat(vars, xNotEqualIndex_Int[i]; dims=1) 
    end
    push!(vars, minusNumberOfExchanges)

    #minusNumberOfExchanges will take the necessary value to compensate the occurences of "1" in xNotEqualIndex_Int
    objective = SeaPearl.SumToZero(vars, trailer)
    SeaPearl.addConstraint!(model, objective)
    SeaPearl.addObjective!(model, minusNumberOfExchanges)

    ### Solve ###
    @time SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return model
end

"""
solve_kidneyexchange_matrix(filename::String)

Return the SeaPearl model solved (solution as a matrix) for to the KEP problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic
This model does allow self-compatible pairs. If donor from pair i gives a kidney to patient of pair i, it will be a size 1 cycle represented as a 1 in the i-th element of the diagonal of the matrix solution.

# Arguments
- `filename`: file containing the compatibilities between pairs

# Constraints
- SumLessThan
- SumToZero

# Objective: maximize the number of valid exchanges

# Branchable variables
- matrix n x n where: 
    n is the reduced number of pairs
    x[i, j] = 1 => pair i receive a kidney from pair j
    x[i, j] = 0 => pair i do not receive a kidney from pair j
"""
function solve_kidneyexchange_matrix(filename::String)
    inputData = getInputData(filename)
    c = inputData.compatibilities
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Attempting to reduce the instance ### 
    canReduce, pairsEquivalence, reduced_c = reduce_instance(inputData)
    if canReduce
        model.adhocInfo = pairsEquivalence
        c = reduced_c
    end
    n = length(c)
    
    ### Variable declaration ###    

    #x[i, j] = 1 => pair i receive a kidney from pair j
    #x[i, j] = 0 => pair i do not receive a kidney from pair j
    x = Matrix{SeaPearl.AbstractIntVar}(undef, n, n)

    #minus_x = x * -1
    #Usefull to check with SumToZero that for each pair: give a kidney <=> receive a kidney
    minus_x = Matrix{SeaPearl.AbstractIntVar}(undef, n, n) 

    for i = 1:n
        for j = 1:n
            if j ∈ c[i]
                x[i, j] = SeaPearl.IntVar(0, 1, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=true)
                minus_x[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            else
                #force x[i, j] = 0 as j ∉ c[i]
                x[i, j] = SeaPearl.IntVar(0, 0, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=false)
                minus_x[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            end
        end
    end

    ### Constraints ###
    for i = 1:n
        #Check that any pair receives more than 1 kidney
        SeaPearl.addConstraint!(model, SeaPearl.SumLessThan(x[i, :], 1, trailer))

        #Check that any pair gives more than 1 kidney
        SeaPearl.addConstraint!(model, SeaPearl.SumLessThan(x[:, i], 1, trailer))

        #Check that for each pair: give a kidney <=> receive a kidney
        SeaPearl.addConstraint!(model, SeaPearl.SumToZero(hcat(x[:, i], minus_x[i, :]), trailer))
    end

    ### Objective ###

    #SeaPearl's solver minimize the objective variable, so we use minusNumberOfExchanges in order to maximize the number of exchanges
    minusNumberOfExchanges = SeaPearl.IntVar(-n, 0, "minusNumberOfExchanges", trailer) 
    SeaPearl.addVariable!(model, minusNumberOfExchanges; branchable=false)
    vars = SeaPearl.AbstractIntVar[]

    #Concatenate all values of x and minusNumberOfExchanges
    for i in 1:n
        vars = cat(vars, x[i, :]; dims=1)
    end
    push!(vars, minusNumberOfExchanges)

    #minusNumberOfExchanges will take the necessary value to compensate the occurences of "1" in x
    objective = SeaPearl.SumToZero(vars, trailer)
    SeaPearl.addConstraint!(model, objective)
    SeaPearl.addObjective!(model, minusNumberOfExchanges)

    ### Solve ###
    @time SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return model
end


"""
reduce_instance(inputData)

Try to remove recursively from compatibilities the pairs that can't receive nor give any kidney
Create a vector (pairsEquivalence) to record the equivalence between the original instance and the reduced version

# Output
- canReduce: bool to know if the instance is reductible
- pairsEquivalence: vector to record the equivalence between the original instance and the reduced version 
- c: reduced version of the vector compatibilities
"""
function reduce_instance(inputData)
    n = inputData.numberOfPairs
    c = inputData.compatibilities
    canReduce = true
    current_n = n

    #Remove recursively from vectors in compatibilities the pairs that can't receive nor give any kidney
    while canReduce
        flatten_c = reduce(vcat,c)
        usefullPairs = [i for i in 1:n if i in flatten_c && c[i] != []] 
        if length(usefullPairs) < current_n
            c = [filter(e -> e in usefullPairs, list) for list in c]
            current_n = length(usefullPairs)
        else
            canReduce = false
        end
    end

    #Check if problem is reduced
    if n != current_n

        #Create a vector (pairsEquivalence) to record the equivalence between the original instance and the reduced version
        pairsEquivalence = []
        for i in 1:n
            if !isempty(c[i])
                push!(pairsEquivalence, i)
            end
        end
    
        #Remove from compatibilities the isolated pairs
        c = filter(e -> !isempty(e), c)

        #Update compatibilities using pairsEquivalence
        for i in 1:length(c)
            c[i] = map(e -> findfirst(x-> x == e, pairsEquivalence), c[i])
        end

        println("Instance reduction: original size "*string(n)*", reduced size "*string(current_n))
        println()
        return true, pairsEquivalence, c
        
    else
        println("Irreductible instance")
        return false, nothing, nothing
    end 
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
        for i in 1:length(pairsEquivalence)
            #Pad with whitespace to align values
            firstPair = string(pairsEquivalence[i])
            formatedFirstPair = firstPair*" "^(length(string(pairsEquivalence[end]))-length(firstPair))
            println(formatedFirstPair*" | "*string(i))
        end
        println()
        println()
end