using SeaPearl

struct Item
    id      :: Int
    value   :: Int
    weight  :: Int
end

mutable struct Solution
    content     :: AbstractArray{Bool}
    value       :: Int
    weight      :: Int
    optimality  :: Bool
end

struct InputData
    items               :: AbstractArray{Union{Item, Nothing}}
    sortedItems         :: AbstractArray{Union{Item, Nothing}}
    numberOfItems       :: Int
    capacity            :: Int
end

include("IOmanager.jl")


function solve_knapsack(filename::String; benchmark=false)
    
    input = parseFile!(filename)

    permutation = sortperm(input.items; by=(x) -> x.value/x.weight, rev=true)

    n = input.numberOfItems

    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Variable declaration ###
    x_s = SeaPearl.IntVar[]
    x_a = SeaPearl.IntVar[]
    for i in 1:n
        push!(x_s, SeaPearl.IntVar(0, input.capacity, "x_s[" * string(i) * "]", trailer))
        push!(x_a, SeaPearl.IntVar(0, 1, "x_a[" * string(i) * "]", trailer))
        SeaPearl.addVariable!(model, last(x_s))
        SeaPearl.addVariable!(model, last(x_a))
    end

    push!(x_s, SeaPearl.IntVar(0, input.capacity, "x_s[" * string(n+1) * "]", trailer))
    SeaPearl.addVariable!(model, last(x_s))


    ### Constraints ###
    # Initial state: x_s[1] = 0
    initial = SeaPearl.EqualConstant(x_s[1], 0, trailer)
    push!(model.constraints, initial)

    # Transition: x_s[i+1] = x_s[i] + w[i]*x_a[i]
    for i in 1:n
        w_x_a_i = SeaPearl.IntVarViewMul(x_a[i], input.items[permutation[i]].weight, "w["*string(i)*"]*x_a["*string(i)*"]")
        minusX_s = SeaPearl.IntVarViewOpposite(x_s[i+1], "-x_s["*string(i+1)*"]")
        SeaPearl.addVariable!(model, w_x_a_i)
        SeaPearl.addVariable!(model, minusX_s)
        vars = SeaPearl.AbstractIntVar[w_x_a_i, minusX_s, x_s[i]]
        transition = SeaPearl.SumToZero(vars, trailer)
        push!(model.constraints, transition)
    end

    ### Objective ### minimize: -sum(v[i]*x_a[i])
    vars = SeaPearl.AbstractIntVar[]
    maxValue = 0
    for i in 1:n
        vx_a_i = SeaPearl.IntVarViewMul(x_a[i], input.items[permutation[i]].value, "v["*string(i)*"]*x_a["*string(i)*"]")
        push!(vars, vx_a_i)
        maxValue += input.items[permutation[i]].value
    end
    y = SeaPearl.IntVar(-maxValue, 0, "y", trailer)
    SeaPearl.addVariable!(model, y)
    push!(vars, y)
    objective = SeaPearl.SumToZero(vars, trailer)
    push!(model.constraints, objective)
    model.objective = y



    status = SeaPearl.solve!(model; variableHeuristic=selectVariable)

    if !benchmark
        print(status)
        for oneSolution in model.solutions
            output = solutionFromSeaPearl(oneSolution, input, permutation)
            printSolution(output)
        end
    end
    return status
end

function solve_knapsack_without_dp(filename::String; benchmark=false)
    input = parseFile!(filename)

    permutation = sortperm(input.items; by=(x) -> x.value/x.weight, rev=true)

    n = input.numberOfItems

    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Variable declaration ###
    x = SeaPearl.IntVar[]
    for i in 1:n
        push!(x, SeaPearl.IntVar(0, 1, "x[" * string(i) * "]", trailer))
        SeaPearl.addVariable!(model, last(x))
    end


    ### Constraints ###

    # Creating the totalWeight variable
    varsWeight = SeaPearl.AbstractIntVar[]
    maxWeight = 0
    for i in 1:n
        wx_i = SeaPearl.IntVarViewMul(x[i], input.items[permutation[i]].weight, "w["*string(i)*"]*x["*string(i)*"]")
        push!(varsWeight, wx_i)
        maxWeight += input.items[permutation[i]].weight
    end
    totalWeight = SeaPearl.IntVar(0, maxWeight, "totalWeight", trailer)
    minusTotalWeight = SeaPearl.IntVarViewOpposite(totalWeight, "-totalWeight")
    SeaPearl.addVariable!(model, totalWeight)
    SeaPearl.addVariable!(model, minusTotalWeight)
    push!(varsWeight, minusTotalWeight)
    weightEquality = SeaPearl.SumToZero(varsWeight, trailer)
    push!(model.constraints, weightEquality)

    # Making sure it is below the capacity
    weightConstraint = SeaPearl.LessOrEqualConstant(totalWeight, input.capacity, trailer)
    push!(model.constraints, weightConstraint)



    ### Objective ### minimize: -sum(v[i]*x_a[i])

    # Creating the sum
    varsValue = SeaPearl.AbstractIntVar[]
    maxValue = 0
    for i in 1:n
        vx_i = SeaPearl.IntVarViewMul(x[i], input.items[permutation[i]].value, "v["*string(i)*"]*x["*string(i)*"]")
        push!(varsValue, vx_i)
        maxValue += input.items[permutation[i]].value
    end
    totalValue = SeaPearl.IntVar(-maxValue, 0, "totalValue", trailer)
    SeaPearl.addVariable!(model, totalValue)
    push!(varsValue, totalValue)
    valueEquality = SeaPearl.SumToZero(varsValue, trailer)
    push!(model.constraints, valueEquality)

    # Setting it as the objective
    model.objective = totalValue



    status = SeaPearl.solve!(model; variableHeuristic=selectVariableWithoutDP)

    if !benchmark
        print(status)
        for oneSolution in model.solutions
            output = solutionFromSeaPearlWithoutDP(oneSolution, input, permutation)
            printSolution(output)
        end
    end
    return status
end

function selectVariable(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables["x_a[" * string(i) * "]"])
        i += 1
    end
    return model.variables["x_a[" * string(i) * "]"]
end

function selectVariableWithoutDP(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables["x[" * string(i) * "]"])
        i += 1
    end
    return model.variables["x[" * string(i) * "]"]
end

function solutionFromSeaPearl(SeaPearlSol::SeaPearl.Solution, input::InputData, permutation::Array{Int})
    taken = falses(input.numberOfItems)
    value = 0
    weight = 0
    for i in 1:input.numberOfItems
        if haskey(SeaPearlSol, "x_a[" * string(i) * "]")
            taken[permutation[i]] = convert(Bool, SeaPearlSol["x_a[" * string(i) * "]"])
            if taken[permutation[i]]
                value += input.items[permutation[i]].value
                weight += input.items[permutation[i]].weight
            end
        end
    end
    return Solution(taken, value, weight, false)
end


function solutionFromSeaPearlWithoutDP(SeaPearlSol::SeaPearl.Solution, input::InputData, permutation::Array{Int})
    taken = falses(input.numberOfItems)
    value = 0
    weight = 0
    for i in 1:input.numberOfItems
        if haskey(SeaPearlSol, "x[" * string(i) * "]")
            taken[permutation[i]] = convert(Bool, SeaPearlSol["x[" * string(i) * "]"])
            if taken[permutation[i]]
                value += input.items[permutation[i]].value
                weight += input.items[permutation[i]].weight
            end
        end
    end
    return Solution(taken, value, weight, false)
end
