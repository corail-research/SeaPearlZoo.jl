using SeaPearl


"""
    solve_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

return the SeaPearl model for to the N-Queens problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent.

# Arguments
- `board_size::Int`: dimension of the board
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function solve_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
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

    return model
end

"""
    print_queens(board_size::Int; nb_sols=typemax(Int), benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Print at max nb_sols solutions to the N-queens problems, N givern as the board_size entry.

# Arguments
- `board_size::Int`: dimension of the board
- 'nb_sols::Int' : maximum number of solutions to print
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function print_queens(board_size::Int; nb_sols=typemax(Int), benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    model = solve_queens(board_size; variableSelection=variableSelection, valueSelection=valueSelection)
    variables = model.variables
    solutions = model.solutions
    count = 0
    println("The solver found "*string(length(solutions))*" solutions to the "*string(board_size)*"-queens problem. Let's show them.")
    println()
    for key in keys(solutions)
        if(count >= nb_sols)
            break
        end
        println("Solution "*string(count+1))
        count +=1
        sol = solutions[key]
        for i in 1:board_size
            ind_queen = sol["row_"*string(i)]
            for j in 1:board_size
                if (j==ind_queen) print("Q ") else print("_ ") end
            end
            println()
        end
        println()
    end
end

"""
    nb_solutions_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())::Int

return the number of solutions for to the N-Queens problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent.

# Arguments
- `board_size::Int`: dimension of the board
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function nb_solutions_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())::Int
    return(length(solve_queens(board_size; variableSelection=variableSelection, valueSelection=valueSelection).solutions))
end
