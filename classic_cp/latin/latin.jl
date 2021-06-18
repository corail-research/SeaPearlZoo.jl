using SeaPearl

struct InputData
    A::Matrix{Int} #number of lines
end

"""
    model_eternity2(input_file; order=[1,2,3,4], limit=nothing)

return the SeaPearl model for to the solve_eternity2 problem using SeaPearl.AllDifferent  and SeaPearl.TableConstraint (without solving it)

# Arguments
- `input_file`: file containing the pieces of the game as well as the dimensions
- 'order' : Vector, giving the order of edges for the IO manager. example : [1,4,2,3] means it is given as [up,down,left,right]
- 'limit' : Int, giving the number of solutions after which it will stop searching. if nothing given, it will lookk for all the solutions
"""
function model_latin(input::InputData; limit=1)
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)
    model.limit.numberOfSolutions = limit
    A = InputData.A
    N = size(A)[1]
    puzzle = Matrix{SeaPearl.AbstractIntVar}(undef, N,N)
    for i = 1:N
        for j in 1:N
            puzzle[i,j] = SeaPearl.IntVar(1, N, "puzzle_"*string(i)*","*string(j), trailer)
            SeaPearl.addVariable!(cpmodel, puzzle[i,j]; branchable=true)
            if A[i,j]>0 push!(model.constraints,SeaPearl.EqualConstant(puzzle[i,j], A[i,j], trailer)) end
        end
    end
    for i in 1:N
        push!(cpmodel.constraints, SeaPearl.AllDifferent(puzzle[i,:], cpmodel.trailer))
        push!(cpmodel.constraints, SeaPearl.AllDifferent(puzzle[:,i], cpmodel.trailer))
    end
    return model
end

function solve_latin!(model::SeaPearl.CPModel; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

function print_latin(model)
    sol= model.statistics.solutions[1]
    n = oftype(1,sqrt(length(sol)))
    tableau = Matrix{Int}(undef,n,n)
    for i in 1:n, j in 1:n
        tableau[i, j]= sol["puzzle_"*string(i)*","*string(j)]
    end
    print(" ")
    for k in 1:5*n
        print("-")
    end
    println()
    for i in 1:n
        print("|")
        for j in 1:n
            printstyled(" "*string(tableau[i,j],pad=2)*" ", color=tableau[i,j])
            print("|")
        end
        println()
    end
    print(" ")
    for k in 1:5*n
        print("-")
    end
end
