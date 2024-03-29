using JuMP
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

struct KnapsackVariableSelectionJuMP <: SeaPearl.AbstractVariableSelection{true} end
function (::KnapsackVariableSelectionJuMP)(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables[string(i)])
        i += 1
    end
    return model.variables[string(i)]
end

function solve_knapsack_JuMP(filename::String; benchmark=false)
    input = parseFile!(filename)

    permutation = sortperm(input.items; by=(x) -> x.value/x.weight, rev=true)

    n = input.numberOfItems

    model = Model(SeaPearl.Optimizer)

    ### Variable declaration ###
    @variable(model, 0 <= x[1:n] <= 1)


    ### Constraints ###
    @expression(model, weight_sum, input.items[permutation[1]].weight * x[1])
    for i in 2:n
        add_to_expression!(weight_sum, input.items[permutation[i]].weight, x[i])
    end
    @constraint(model, 0 <= weight_sum <= input.capacity)



    ### Objective ### minimize: -sum(v[i]*x_a[i])
    @expression(model, val_sum, input.items[permutation[1]].value * x[1])
    for i in 2:n
        add_to_expression!(val_sum, input.items[permutation[i]].value, x[i])
    end
    @objective(model, Min, -val_sum)


    # define the heuristic used for variable selection
    variableheuristic = KnapsackVariableSelectionJuMP()
    MOI.set(model, SeaPearl.MOIVariableSelectionAttribute(), variableheuristic)

    optimize!(model)
    status = MOI.get(model, MOI.TerminationStatus())

    println(model)
    println(status)
    println(has_values(model))
    println(value.(x))
    println(solutionFromJuMPWithoutDP(value.(x), input, permutation))
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


function solutionFromJuMPWithoutDP(jump_sol, input::InputData, permutation::Array{Int})
    taken = falses(input.numberOfItems)
    value = 0
    weight = 0
    for i in 1:input.numberOfItems
        taken[permutation[i]] = convert(Bool, jump_sol[i])
        if taken[permutation[i]]
            value += input.items[permutation[i]].value
            weight += input.items[permutation[i]].weight
        end
    end
    return Solution(taken, value, weight, false)
end
