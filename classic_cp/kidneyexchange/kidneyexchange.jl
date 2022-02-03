using SeaPearl
struct InputData
    numberOfPairs         :: Int
    density               :: Float16
    compatibilities       :: Vector{Vector{Int}}
end

include("IOmanager.jl")

"""
    Unfinished version, problem defining the objective (how to convert IntVar to Bool?)
"""
function solve_kidneyexchange_bis(filename::String)

    InputData = getInputData(filename)

    n = InputData.numberOfPairs
    c = InputData.compatibilities

    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Try to reduce problem ###    
    canReduce = true
    current_n = n
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

    pairsEquivalence = []
    for i in 1:n
        if !isempty(c[i])
            push!(pairsEquivalence, i)
        end
    end
    findfirst(x-> x == pair[1], pairsEquivalence)

    c = filter(e -> !isempty(e), c)

    for i in 1:length(c)
        c[i] = map(e -> findfirst(x-> x == e, pairsEquivalence), c[i])
        push!(c[i], i)
    end

    println("Instance reduction: original size "*string(n)*", reduced size "*string(current_n))
    println()
    n = current_n
    
    ### Variable declaration ###    
    x = Vector{SeaPearl.AbstractIntVar}(undef, n) 
    x_offset = Vector{SeaPearl.AbstractIntVar}(undef, n) 

    for i = 1:n
        x[i] = SeaPearl.IntVar(1, n, "x_"*string(i), model.trailer)
        SeaPearl.addVariable!(model, x[i]; branchable=true)
        x_offset[i] = SeaPearl.IntVarViewOffset(x[i], -i, x[i].id*"-"*string(i))
    end

    ### Constraints ###
    for i = 1:n
        for j in 1:n
            if j ∉ c[i]
                SeaPearl.addConstraint!(model, SeaPearl.NotEqualConstant(x[i], j, trailer))
            end
        end
    end
    SeaPearl.addConstraint!(model, SeaPearl.AllDifferent(x, trailer))

    # additional constraint: sum(x_offset) = 0

    ### Objective ###
    numberOfExchanges = SeaPearl.IntVar(-n, 0, "numberOfExchanges", trailer) 
    SeaPearl.addVariable!(model, numberOfExchanges; branchable=true)
    vars = SeaPearl.AbstractIntVar[]
    for i in 1:n
        vars = cat(vars, int2bool(x_offset[i]); dims=1) #TODO int2bool?
    end
    push!(vars, numberOfExchanges)
    objective = SeaPearl.SumToZero(vars, trailer)
    SeaPearl.addConstraint!(model, objective)
    SeaPearl.addObjective!(model, numberOfExchanges)

    ### Solve ###
    @time SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return model

end

"""
solve_kidneyexchange(filename::String)

Return the SeaPearl model solved for to the KEP problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic

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
function solve_kidneyexchange(filename::String)

    InputData = getInputData(filename)
    n = InputData.numberOfPairs
    c = InputData.compatibilities
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Try to reduce problem ###    
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
        model.adhocInfo = pairsEquivalence
    
        #Remove from compatibilities the isolated pairs
        c = filter(e -> !isempty(e), c)

        #Update compatibilities using pairsEquivalence
        for i in 1:length(c)
            c[i] = map(e -> findfirst(x-> x == e, pairsEquivalence), c[i])
        end

        println("Instance reduction: original size "*string(n)*", reduced size "*string(current_n))
        println()
        n = current_n
    else
        println("Irreductible instance")
    end
    
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
print_solutions(solved_model::SeaPearl.CPModel)

Print the optimal solution (in matrix form and as a set of cycles) calculated by solve_kidneyexchange()

# Print format
- matrix form
    m[i, j] = 1 => pair i receive a kidney from pair j
    m[i, j] = 0 => pair i do not receive a kidney from pair j
- set of cycles form
    4 -> 7 -> 1 => pair 4 gives a kidney to pair 7, pair 7 gives a kidney to pair 1 and pair 1 gives a kidney to pair 4
"""

function print_solutions(solved_model::SeaPearl.CPModel)

    #Filter solutions to remove "nothing" and non-optimal solutions
    solutions = solved_model.statistics.solutions
    if isdefined(solved_model, :adhocInfo)
        isReduced = true
        pairsEquivalence = solved_model.adhocInfo
    else
        isReduced = false
    end
    numberOfPairs = trunc(Int, sqrt(length(solved_model.variables) - 1))
    count = 0
    realSolutions = filter(e -> !isnothing(e),solutions)
    bestScores = map(e -> -minimum(values(e)),realSolutions)
    bestScore = maximum(bestScores)
    bestSolution = filter(e -> -minimum(values(e)) == bestScore, realSolutions)
    println("The solver found an optimal solutions with "*string(bestScore)*" exchanges to the KEP problem. Let's show it.")
    println()

    for sol in bestSolution
        
        #Print matrix
        print("Solution as a matrix")
        if isReduced print(" (reduced instance)") end
        println()
        coordOnes = []
        count +=1
        for i in 1:numberOfPairs
            for j in 1:numberOfPairs
                val = string(sol["x_"*string(i)*"_"*string(j)])
                print(val*" ")
                if val == "1"
                    push!(coordOnes, (i, j))
                end
            end
            println()
        end
        println()

        #Find cycles
        print("Solution as a set of cycles")
        if isReduced print(" (original instance)") end
        println()
        cycles = []
        while !isempty(coordOnes)
            current = pop!(coordOnes)
            cycle = []
            push!(cycle, current)
            if current[1] != current[2]
                isOpen = true
            else
                #Edge case: pair i is compatible with itself
                isOpen = false
            end

            while isOpen
                idx = findfirst(x -> x[1] == cycle[end][2], coordOnes)
                current = splice!(coordOnes, idx)
                push!(cycle, current)
                if cycle[1][1] == cycle[end][2]
                    isOpen = false
                end
            end
            push!(cycles, cycle)
        end
        
        #Print cycles
        for cycle in cycles
            print("Cycle of size "*string(length(cycle))*": ")
            for pair in cycle
                isReduced ? print(string(pairsEquivalence[pair[1]])) : print(string(pair[1]))
                if pair != cycle[end]
                    print(" -> ")
                end
            end
            println("\n")
        end

        #Print pairsEquivalence (link between the original pairs and the reduced pairs)
        if isReduced
            println("Table of equivalence: Original pair | Reduced pair")
            for i in 1:length(pairsEquivalence)
                #Pad with whitespace to align values
                firstPair = string(pairsEquivalence[i])
                formatedFirstPair = firstPair*" "^(length(string(pairsEquivalence[end]))-length(firstPair))
                println(formatedFirstPair*" | "*string(i))
            end
        end
    end
end