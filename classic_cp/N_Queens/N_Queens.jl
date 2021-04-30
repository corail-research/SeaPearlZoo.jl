using SeaPearl

"""
    solve_queens(board_size::int; benchmark=false)::Int

Find the number of solutions to the N-Queens problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent.

# Arguments
- `board_size::Int`: dimension of the board
"""
function solve_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())::Int
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    rows = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows[i] = SeaPearl.IntVar(1, board_size, "row_"*string(i), trailer)
        SeaPearl.addVariable!(model, rows[i]; branchable=true)
    end

    rows_plus = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows_plus[i] = SeaPearl.IntVarViewOffset(rows[i], i, rows[i].id*"+"*string(i))
    end

    rows_minus = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows_minus[i] = SeaPearl.IntVarViewOffset(rows[i], -i, rows[i].id*"-"*string(i))
    end

    push!(model.constraints, SeaPearl.AllDifferent(rows, trailer))
    push!(model.constraints, SeaPearl.AllDifferent(rows_plus, trailer))
    push!(model.constraints, SeaPearl.AllDifferent(rows_minus, trailer))

    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)

    return length(model.solutions)
end
