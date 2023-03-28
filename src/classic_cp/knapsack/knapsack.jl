using SeaPearl
include("knapsackIOmanager.jl")

"""build_model_and_solve_knapsack(data::KnapsackInputData; benchmark=false)
builds knapsack model based on input data provided and returns model

"""
function build_model_and_solve_knapsack(data::KnapsackInputData; benchmark=false)::SeaPearl.CPModel
    sorted_relative_value = sortperm(data.items; by=(x) -> x.value/x.weight, rev=true) # sort items by relative value
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)
    model = build_knapsack_model!(data, sorted_relative_value, model, trailer)
    status = SeaPearl.solve!(model; variableHeuristic=VariableSelection{true}())
    
    return model
end

"""
    build_knapsack_model!(data::KnapsackInputData, sorted_relative_value::Vector{Int64}, model::SeaPearl.CPModel, trailer::SeaPearl.Trailer)
Returns a model with variables, constraints and objective of a knapsack problem, defined by the contents of the "data" argument

# Args:
- data:: KnapsackInputData. Data loaded from sample datasets available in SeaPearlZoo/classic_cp/knapsack/data. The helper function parseFile!
  is useful to load these datasets in the proper format 
- sorted_relative_value:: Vector{Int64}. Contains the index of the items in data ^, after they were sorted by relative weight 
- model::CPModel. empty model that will be built 
- trailer::SeaPearl.Trailer. trailer
"""
function build_knapsack_model!(data::KnapsackInputData, sorted_relative_value::Vector{Int64}, model::SeaPearl.CPModel, trailer::SeaPearl.Trailer)
    num_items = data.numberOfItems
    # =========VARIABLES=========
    # add variables representing item selection
    item_selection = SeaPearl.IntVar[]
    for i in 1: num_items
        push!(item_selection, SeaPearl.IntVar(0, 1, "item[" * string(i) * "]", trailer))
        SeaPearl.addVariable!(model, last(item_selection))
    end
    # add variables representing the weight of the item in the knapsack. If the item is not selected, the variable associated to this item will have 
    # a value of zero
    item_weight_in_knapsack = SeaPearl.AbstractIntVar[]
    max_weight = 0
    for i in 1:num_items
        current_item_index = sorted_relative_value[i]
        current_item = data.items[current_item_index]
        current_item_weight_in_knapsack = SeaPearl.IntVarViewMul(item_selection[i], current_item.weight, "weight_item["*string(i)*"]")
        push!(item_weight_in_knapsack, current_item_weight_in_knapsack)
        max_weight += current_item.weight
    end
    total_weight = SeaPearl.IntVar(0, max_weight, "total_weight", trailer) # total weight of items in knapsack
    negative_total_weight = SeaPearl.IntVarViewOpposite(total_weight, "-total_weight") # -1 * total_weight; necessary because the solver can only minimize objectives
    SeaPearl.addVariable!(model, total_weight)
    SeaPearl.addVariable!(model, negative_total_weight)
    push!(item_weight_in_knapsack, negative_total_weight) # added to array to later add a constraint that the sum of the array's elements == 0

    item_value_in_knapsack = SeaPearl.AbstractIntVar[]
    max_value = 0
    for i in 1:num_items
        current_item_index = sorted_relative_value[i]
        current_item = data.items[current_item_index]
        current_item_value = SeaPearl.IntVarViewMul(item_selection[i], current_item.value, "value_item["*string(i)*"]")
        push!(item_value_in_knapsack, current_item_value)
        max_value += current_item.value
    end
    total_value = SeaPearl.IntVar(-max_value, 0, "totalValue", trailer)
    SeaPearl.addVariable!(model, total_value)
    push!(item_value_in_knapsack, total_value)

    # =========CONSTRAINTS=========
    
    # consistency of negative weight in knapsack
    weight_sums_to_zero = SeaPearl.SumToZero(item_weight_in_knapsack, trailer)
    push!(model.constraints, weight_sums_to_zero)
    # weight below max capacity
    weight_constraint = SeaPearl.LessOrEqualConstant(total_weight, data.capacity, trailer) 
    push!(model.constraints, weight_constraint)
    # consistency of negative value variable in knapsack
    valueEquality = SeaPearl.SumToZero(item_value_in_knapsack, trailer)
    push!(model.constraints, valueEquality)

    # =========OBJECTIVE=========
    SeaPearl.addObjective!(model, total_value)
    
    return model
end

struct VariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
VariableSelection(;take_objective=true) = VariableSelection{take_objective}() # question: take_objective ???

"""(::VariableSelection{true})(model::SeaPearl.CPModel)::SeaPearl.AbstractIntVar


"""
function (::VariableSelection{true})(model::SeaPearl.CPModel)::SeaPearl.AbstractIntVar
    i = 1
    while SeaPearl.isbound(model.variables["item[" * string(i) * "]"])
        i += 1
    end
    return model.variables["item[" * string(i) * "]"]
end

input_data = parseKnapsackFile!("./data/ks_4_0")
model = build_model_and_solve_knapsack(input_data)
