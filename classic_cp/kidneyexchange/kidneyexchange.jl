using SeaPearl

"""
Instances format
4 0.33 #numberOfPairs density
2 4    #pair 1 can receive a kidney from pairs 2 and 4
1      #pair 2 can receive a kidney from pair 1
4      #pair 3 can receive a kidney from pair 4
3      #pair 4 can receive a kidney from pair 3
"""

struct InputData
    numberOfPairs         :: Int
    density               :: Float16
    compatibilities       :: Vector{Union{Vector{Int}, Nothing}}
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

    count = 1
    pairsEquivalence = []
    for i in 1:n
        if isempty(c[i])
            push!(pairsEquivalence, nothing)
        else
            push!(pairsEquivalence, count)
            count += 1
        end
    end

    c = filter(e -> !isempty(e), c)

    for i in 1:length(c)
        c[i] = map(e -> pairsEquivalence[e], c[i])
        push!(c[i], i)
    end

    println("Instance reduction: original size "*string(n)*", new size "*string(current_n))
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
                push!(model.constraints, SeaPearl.NotEqualConstant(x[i], j, trailer))
            end
        end
    end
    push!(model.constraints, SeaPearl.AllDifferent(x, trailer))

    # additional constraint: sum(x - x_offset) = 0

    ### Objective ###
    numberOfExchanges = SeaPearl.IntVar(-n, 0, "numberOfExchanges", trailer) 
    SeaPearl.addVariable!(model, numberOfExchanges)
    vars = SeaPearl.AbstractIntVar[]
    for i in 1:n
        vars = cat(vars, int2bool(x_offset[i]); dims=1) #TODO int2bool?
    end
    push!(vars, numberOfExchanges)
    objective = SeaPearl.SumToZero(vars, trailer)
    push!(model.constraints, objective)
    SeaPearl.addObjective!(model, numberOfExchanges)

    ### Solve ###
    @time SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return model

end

"""
solve_kidneyexchange(filename::String)

return the SeaPearl model solved for to the KEP problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic

Constraints used: SumLessThan and SumToZero
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

    count = 1
    pairsEquivalence = []
    for i in 1:n
        if isempty(c[i])
            push!(pairsEquivalence, nothing)
        else
            push!(pairsEquivalence, count)
            count += 1
        end
    end

    c = filter(e -> !isempty(e), c)

    for i in 1:length(c)
        c[i] = map(e -> pairsEquivalence[e], c[i])
    end

    println("Instance reduction: original size "*string(n)*", new size "*string(current_n))
    println()
    n = current_n
    
    ### Variable declaration ###    
    x = Matrix{SeaPearl.AbstractIntVar}(undef, n, n) 
    minusx = Matrix{SeaPearl.AbstractIntVar}(undef, n, n) 

    for i = 1:n
        for j = 1:n
            if j ∈ c[i]
                x[i, j] = SeaPearl.IntVar(0, 1, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=true)
                minusx[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            else
                x[i, j] = SeaPearl.IntVar(0, 0, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=true)
                minusx[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            end
        end
    end

    ### Constraints ###
    for i = 1:n
        push!(model.constraints, SeaPearl.SumLessThan(x[i, :], 1, trailer))
        push!(model.constraints, SeaPearl.SumLessThan(x[:, i], 1, trailer))
        push!(model.constraints, SeaPearl.SumToZero(hcat(x[:, i], minusx[i, :]), trailer))
    end

    ### Objective ###
    numberOfExchanges = SeaPearl.IntVar(-n, 0, "numberOfExchanges", trailer) 
    SeaPearl.addVariable!(model, numberOfExchanges)
    vars = SeaPearl.AbstractIntVar[]
    for i in 1:n
        vars = cat(vars, x[i, :]; dims=1)
    end
    push!(vars, numberOfExchanges)
    objective = SeaPearl.SumToZero(vars, trailer)
    push!(model.constraints, objective)
    SeaPearl.addObjective!(model, numberOfExchanges)

    ### Solve ###
    @time SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return model
end

"""
Solutions format
m[i, j] means that pair i receive a kidney from pair j
"""

function print_solutions(model::SeaPearl.CPModel; nb_sols=typemax(Int))
    solutions = model.statistics.solutions
    numberOfPairs = trunc(Int, sqrt(length(model.variables) - 1))
    count = 0
    realSolutions = filter(e -> !isnothing(e),solutions)
    bestScores = map(e -> -minimum(values(e)),realSolutions)
    bestScore = maximum(bestScores)
    bestSolutions = filter(e -> -minimum(values(e)) == bestScore, realSolutions)
    println("The solver found "*string(length(bestSolutions))*" optimal solutions ("*string(bestScore)*" exchanges) to the KEP problem. Let's show them.")
    println()
    for sol in bestSolutions
        if(count >= nb_sols)
            break
        end
        coordOnes = []
        println("Solution "*string(count+1))
        println()

        #Print matrix
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

        #Print cycles
        cycles = []
        while !isempty(coordOnes)
            current = pop!(coordOnes)
            cycle = []
            push!(cycle, current)
            if current[1] != current[2]
                isOpen = true
            else
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
        
        for c in cycles
            print("Cycle of size "*string(length(c))*": ")
            for p in c
                print(string(p[1]))
                if p != c[end]
                    print(" -> ")
                end
            end
            println()
            println()
        end
    end
end



