using SeaPearl

struct Queen
    id::Int
    pos::Int
end

struct Problem
    Queens::AbstractArray{Queen}
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