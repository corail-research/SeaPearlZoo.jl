using SeaPearl

#

struct InputData
    A::Matrix{Int} #number of lines
end

"""
    model_latin(input_file; limit=1)

return the SeaPearl model for to the Latin completion problem using SeaPearl.AllDifferent
The input file can come from the Latin DataGen in SeaPearl.

# Arguments
- `InputData`: Matrix containting the uncompleted game
- 'limit' : Int, giving the number of solutions after which it will stop searching. if nothing given, it will looks for all the solutions
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

"""
    solve_latin!(model::SeaPearl.CPModel; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Solve the SeaPearl model for to the Latin Completion problem, using an existing model

# Arguments
- `model::SeaPearl.CPModel`: model (from model_latin)
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}().
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic().
"""
function solve_latin!(model::SeaPearl.CPModel; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

"""
    print_latin(model::SeaPearl.CPModel)

Nice print the latin game when solved

# Arguments
- `model::SeaPearl.CPModel`: model (from model_latin)
"""
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
