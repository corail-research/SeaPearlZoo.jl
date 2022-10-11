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

solved_model = solve_kidneyexchange_vector("./data/kep_8_0.2")
print_solutions_vector(solved_model)