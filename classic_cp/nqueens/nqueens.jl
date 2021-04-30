using SeaPearl

struct Queen
    id::Int
    pos::Int
end

struct Problem
    Queens::AbstractArray{Queen}
end

"""
    struct MostCenteredVariableSelection{TakeObjective} 

New variableSelection heuristic that selects the legal (ie. among the not bounded ones) most centered Queen 
"""
struct MostCenteredVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
###constructor###
MostCenteredVariableSelection(;take_objective = true) = MostCenteredVariableSelection{take_objective}()

function (::MostCenteredVariableSelection{false})(cpmodel::SeaPearl.CPModel)
    selectedVar = nothing
    n = length(cpmodel.variables)
    #print(SeaPearl.branchable_variables(cpmodel))
    #print(values(cpmodel.variables))
    sorted_dict = sort(collect(SeaPearl.branchable_variables(cpmodel)),by = x -> ceil(n/2.0)-parse(Int64,x[1][5]),rev=true)
    while !isempty(sorted_dict)
        selectedVar= pop!(sorted_dict)[2]
        selectedVar == cpmodel.objective || SeaPearl.isbound(selectedVar) || break
    end

    if SeaPearl.isnothing(selectedVar) && !SeaPearl.isbound(cpmodel.objective)
        return cpmodel.objective
    end
    return selectedVar

end

function (::MostCenteredVariableSelection{true})(cpmodel::SeaPearl.CPModel)
    selectedVar = nothing
    n = length(cpmodel.variables)
    sorted_dict = sort(collect(SeaPearl.branchable_variables(cpmodel)),by = x -> ceil(n/2.0)-parse(Int64,x[1][5]),rev=true)
    while true
        selectedVar= pop!(sorted_dict)[2]
        !SeaPearl.isbound(selectedVar) || break
    end

    return selectedVar

end


function create_nqueens_model(n::Int)


    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    rows = Vector{SeaPearl.AbstractIntVar}(undef, n)
    for i in 1:n
        rows[i] = SeaPearl.IntVar(1, n, "row_" * string(i), trailer)
        SeaPearl.addVariable!(model, rows[i]; branchable = true)
    end

    rows_plus = Vector{SeaPearl.AbstractIntVar}(undef, n)
    for i in 1:n
        rows_plus[i] = SeaPearl.IntVarViewOffset(rows[i], i, rows[i].id * "+" * string(i))
        # SeaPearl.addVariable!(model, rows_plus[i]; branchable=false)
    end

    rows_minus = Vector{SeaPearl.AbstractIntVar}(undef, n)
    for i in 1:n
        rows_minus[i] = SeaPearl.IntVarViewOffset(rows[i], -i, rows[i].id * "-" * string(i))
        # SeaPearl.addVariable!(model, rows_minus[i]; branchable=false)
    end

    push!(model.constraints, SeaPearl.AllDifferent(rows, trailer))
    push!(model.constraints, SeaPearl.AllDifferent(rows_plus, trailer))
    push!(model.constraints, SeaPearl.AllDifferent(rows_minus, trailer))

    variableSelection =  SeaPearl.MostCenteredVariableSelection{false}()
    return model, variableSelection
end

function solve(model::SeaPearl.CPModel, variableSelection::SeaPearl.AbstractVariableSelection{false} )
    status = @time SeaPearl.solve!(model; variableHeuristic = variableSelection)
end