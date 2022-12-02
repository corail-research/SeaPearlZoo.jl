using SeaPearl
include("kidney_exchange_IOmanager.jl")

"""
solve_kidney_exchange_matrix(inputData::KidneyExchangeInputData)

Return the SeaPearl model solved (solution as a matrix) for to the KEP problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic
This model does allow self-compatible pairs. If donor from pair i gives a kidney to patient of pair i, it will be a size 1 cycle represented as a 1 in the i-th element of the diagonal of the matrix solution.

# Arguments
- `inputData`: parsed input data file 

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
function solve_kidney_exchange_matrix(inputData::KidneyExchangeInputData)
    compatibilities = inputData.compatibilities
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Attempting to reduce the instance ### 
    canReduce, pairsEquivalence, reduced_compatibilities = reduce_instance(inputData)
    if canReduce
        model.adhocInfo = pairsEquivalence
        compatibilities = reduced_compatibilities
    end
    num_pairs = length(compatibilities)
    
    # =========VARIABLES=========
    #x[i, j] = 1 => pair i receive a kidney from pair j
    #x[i, j] = 0 => pair i do not receive a kidney from pair j
    x = Matrix{SeaPearl.AbstractIntVar}(undef, num_pairs, num_pairs)

    #minus_x = x * -1
    #Usefull to check with SumToZero that for each pair: give a kidney <=> receive a kidney
    minus_x = Matrix{SeaPearl.AbstractIntVar}(undef, num_pairs, num_pairs) 

    for i = 1:num_pairs
        for j = 1:num_pairs
            if j ∈ compatibilities[i]
                x[i, j] = SeaPearl.IntVar(0, 1, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=true)
                minus_x[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            else
                #force x[i, j] = 0 as j ∉ compatibilities[i]
                x[i, j] = SeaPearl.IntVar(0, 0, "x_"*string(i)*"_"*string(j), model.trailer)
                SeaPearl.addVariable!(model, x[i, j]; branchable=false)
                minus_x[i, j] = SeaPearl.IntVarViewOpposite(x[i, j], "-x_"*string(i)*"_"*string(j))
            end
        end
    end

    # =========CONSTRAINTS=========
    for i = 1:num_pairs
        SeaPearl.addConstraint!(model, SeaPearl.SumLessThan(x[i, :], 1, trailer)) #Check that any pair receives more than 1 kidney
        SeaPearl.addConstraint!(model, SeaPearl.SumLessThan(x[:, i], 1, trailer)) #Check that any pair gives more than 1 kidney
        SeaPearl.addConstraint!(model, SeaPearl.SumToZero(hcat(x[:, i], minus_x[i, :]), trailer)) #Check that for each pair: give a kidney <=> receive a kidney
    end

    # =========OBJECTIVE=========
    minusNumberOfExchanges = SeaPearl.IntVar(-num_pairs, 0, "minusNumberOfExchanges", trailer) # SeaPearl minimizes the objective, so we min the negative objective 
    SeaPearl.addVariable!(model, minusNumberOfExchanges; branchable=false)
    vars = SeaPearl.AbstractIntVar[]

    for i in 1:num_pairs
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
    num_pairs = inputData.numberOfPairs
    compatibilities = inputData.compatibilities
    canReduce = true
    current_num_pairs = num_pairs

    #Remove recursively from vectors in compatibilities the pairs that can't receive nor give any kidney
    while canReduce
        flattened_compatibilities = reduce(vcat,compatibilities)
        usefulPairs = [i for i in 1:num_pairs if i in flattened_compatibilities && compatibilities[i] != []] 
        if length(usefulPairs) < current_num_pairs
            compatibilities = [filter(e -> e in usefulPairs, list) for list in compatibilities]
            current_num_pairs = length(usefulPairs)
        else
            canReduce = false
        end
    end

    if num_pairs != current_num_pairs # problem has been reduced
        pairsEquivalence = [] # Equivalence between the original instance and the reduced version
        for i in 1:num_pairs
            if !isempty(compatibilities[i])
                push!(pairsEquivalence, i)
            end
        end
        compatibilities = filter(e -> !isempty(e), compatibilities) #Remove from compatibilities the isolated pairs
        
        for i in 1: length(compatibilities) #Update compatibilities using pairsEquivalence
            compatibilities[i] = map(e -> findfirst(x-> x == e, pairsEquivalence), compatibilities[i])
        end
        println("Instance reduction: original size "*string(n)*", reduced size "*string(current_num_pairs))
        return true, pairsEquivalence, compatibilities
    else
        println("Irreducible instance")
        return false, nothing, nothing
    end 
end

if abspath(PROGRAM_FILE) == @__FILE__
    inputData = getKidneyExchangeInputData("./data/kep_8_0.2")
    solved_model = solve_kidney_exchange_matrix(inputData)
    print_solutions_matrix(solved_model)
end