using SeaPearl

struct OutputDataQueens
    nb_sols ::Int
    indices ::Matrix{Int}
end

"""
    struct MostCenteredVariableSelection{TakeObjective}

VariableSelection heuristic that selects the legal (ie. among the not bounded ones) most centered Queen.
"""
struct MostCenteredVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
MostCenteredVariableSelection(;take_objective=true) = MostCenteredVariableSelection{take_objective}()

function (::MostCenteredVariableSelection{false})(cpmodel::SeaPearl.CPModel)::SeaPearl.AbstractIntVar
    selected_variable = nothing
    num_variables = length(cpmodel.variables)
    branchable_variables = collect(SeaPearl.branchable_variables(cpmodel))
    
    # sorted_variables will be of type Vector{Pair{String, SeaPearl.AbstractVar}}
    # all elements of the sorted_variables Vector will contain the variable name in position 1
    # and the variable in position 2
    sorted_variables = sort(
        branchable_variables,
        by=x -> get_centered_score(num_variables, x),
        rev=true
    )
    # Loop until an unbound variable is found
    while !isempty(sorted_variables)
        selected_variable = pop!(sorted_variables)[2] # as mentionned above, the second element is the variable
        if !(selected_variable == cpmodel.objective) && !SeaPearl.isbound(selected_variable)
            break
        end
    end

    if SeaPearl.isnothing(selected_variable) && !SeaPearl.isbound(cpmodel.objective)
        return cpmodel.objective
    end
    return selected_variable
end

function (::MostCenteredVariableSelection{true})(cpmodel::SeaPearl.CPModel)::SeaPearl.AbstractIntVar # question: argument{true} ou {false} ?
    selected_variable = nothing
    num_variables = length(cpmodel.variables)
    branchable_variables = collect(SeaPearl.branchable_variables(cpmodel))
    sorted_variables = sort(
        branchable_variables,
        by = x -> get_centered_score(num_variables, x),
        rev=true
    )
    # Loop until an unbound variable is found
    while true
        selected_variable= pop!(sorted_variables)[2]
        if !SeaPearl.isbound(selected_variable)
            break
        end
    end

    return selected_variable
end
"""
    get_centered_score(num_variables::Int, branchable_variable::Pair{String, SeaPearl.AbstractVar})::Float64
Returns the centered score of a row; i.e. how close the queen is to the center of the board.
"""
function get_centered_score(num_variables::Int, branchable_variable::Pair{String, SeaPearl.AbstractVar})::Float64
    variable_name::String = branchable_variable[1]
    row_id::Int = parse(Int, match(r"[0-9]*$", variable_name).match)
    centered_score::Float64 = abs(num_variables / 2 - row_id)
    
    return centered_score
end

"""
    model_queens(board_size::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

return the SeaPearl model for to the N-Queens problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent (without solving it)

# Arguments
- `board_size::Int`: dimension of the board
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}() can be set to MostCenteredVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function model_queens(board_size::Int)
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    # rows[i] designates the row of queen in column i
    rows = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows[i] = SeaPearl.IntVar(1, board_size, "row_"*string(i), trailer)
        SeaPearl.addVariable!(model, rows[i]; branchable=true)
    end
    # diagonals from top left to bottom right
    rows_plus = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows_plus[i] = SeaPearl.IntVarViewOffset(rows[i], i, rows[i].id*"+"*string(i))
    end
    # diagonals top right to bottom left
    rows_minus = Vector{SeaPearl.AbstractIntVar}(undef, board_size)
    for i = 1:board_size
        rows_minus[i] = SeaPearl.IntVarViewOffset(rows[i], -i, rows[i].id*"-"*string(i))
    end

    push!(model.constraints, SeaPearl.AllDifferent(rows, trailer)) # All rows and columns are different - since rows are all different and queens are on different rows
    push!(model.constraints, SeaPearl.AllDifferent(rows_plus, trailer)) 
    push!(model.constraints, SeaPearl.AllDifferent(rows_minus, trailer)) 

    return model
end

"""
    solve_queens(model::SeaPearl.CPModel)

Solve the SeaPearl model for to the N-Queens problem, using an existing model

# Arguments
- `model::SeaPearl.CPModel`: model (from model_queens)
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}().
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic().
"""
function solve_queens(model::SeaPearl.CPModel; variableSelection=MostCenteredVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

"""
    print_queens(model::SeaPearl.CPModel; nb_sols=typemax(Int))

Print at max nb_sols solutions to the N-queens problems.

# Arguments
- `model::SeaPearl.CPModel`: needs the model to be already solved (by solve_queens)
- 'nb_sols::Int' : maximum number of solutions to print
"""
function print_queens(model::SeaPearl.CPModel; nb_sols=typemax(Int))
    variables = model.variables
    solutions = model.statistics.solutions
    board_size = length(model.variables)
    count = 0
    real_solutions = filter(e -> !isnothing(e),solutions)
    println("The solver found "*string(length(real_solutions))*" solutions to the "*string(board_size)*"-queens problem. Let's show them.")
    println()
    for key in keys(real_solutions)
        if(count >= nb_sols)
            break
        end
        sol = real_solutions[key]
        println("Solution "*string(count+1))
        count +=1
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
function nb_solutions_queens(board_size::Int; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())::Int
    return(length(solve_queens(board_size; variableSelection=variableSelection, valueSelection=valueSelection).statistics.solutions))
end

# base_model = model_queens(5)
# solved_model = solve_queens(base_model)
# print_queens(solved_model)
