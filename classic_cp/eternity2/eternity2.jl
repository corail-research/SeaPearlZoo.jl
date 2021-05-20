using SeaPearl

"""
Proposition for using SeaPearl CP solver for the Eternity II puzzle
Reference https://en.wikipedia.org/wiki/Eternity_II_puzzle
"""

include("IOmanager.jl")

struct InputData
    n    ::Int
    m    ::Int
    pieces ::Matrix{Int}
end

struct OutputDataEternityII
    nb_sols ::Int
    orientation::Array{Int, 4}# dims = (nb_sols, n, m, 5) where five corresponds to (id, u,r, d, l)
end


"""
    model_eternity2(input_file; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

return the SeaPearl model for to the solve_eternity2 problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent  and SeaPearl.TableConstraint (without solving it)

# Arguments
- `input_file`: file containing the pieces of the game as well as the dimensions
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
-'order' : Vector
"""
function model_eternity2(input_file; order=[1,2,3,4], variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    inputData = getInputData(input_file;order=order)
    n = inputData.n
    m = inputData.m
    pieces = inputData.pieces

    table = Matrix{Int}(undef, 5, 4*n*m)

    maxi = maximum(pieces)

    for i = 1:n*m
        for k = 1:4
            table[1,4*(i-1) + k] = i
            for j = 2:5
                table[j,4*(i-1) + k] = pieces[i, (j+k+1)%4+1]
            end
        end
    end


    id = Matrix{SeaPearl.AbstractIntVar}(undef, n, m)
    u = Matrix{SeaPearl.AbstractIntVar}(undef, n, m)#up
    r = Matrix{SeaPearl.AbstractIntVar}(undef, n, m)#right
    d = Matrix{SeaPearl.AbstractIntVar}(undef, n, m)#down
    l = Matrix{SeaPearl.AbstractIntVar}(undef, n, m)#left

    for i = 1:n, j=1:m
        id[i,j] = SeaPearl.IntVar(1, n*m, "id_"*string(i)*string(j), trailer)
        SeaPearl.addVariable!(model, id[i,j]; branchable=true)
        u[i,j] = SeaPearl.IntVar(0, maxi, "u_"*string(i)*string(j), trailer)
        SeaPearl.addVariable!(model, u[i,j]; branchable=false)
        r[i,j] = SeaPearl.IntVar(0, maxi, "r_"*string(i)*string(j), trailer)
        SeaPearl.addVariable!(model, r[i,j]; branchable=false)
        d[i,j] = SeaPearl.IntVar(0, maxi, "d_"*string(i)*string(j), trailer)
        SeaPearl.addVariable!(model, d[i,j]; branchable=false)
        l[i,j] = SeaPearl.IntVar(0, maxi, "l_"*string(i)*string(j), trailer)
        SeaPearl.addVariable!(model, l[i,j]; branchable=false)

        vars = SeaPearl.AbstractIntVar[id[i,j], u[i,j], r[i,j],d[i,j], l[i,j]]
        push!(model.constraints, SeaPearl.TableConstraint(vars, table, trailer))

        if (j==m) push!(model.constraints, SeaPearl.EqualConstant(r[i,j], 0, trailer)) end
        if (j==1) push!(model.constraints, SeaPearl.EqualConstant(l[i,j], 0, trailer)) end
        if (i==1) push!(model.constraints, SeaPearl.EqualConstant(u[i,j], 0, trailer)) end
        if (i==n) push!(model.constraints, SeaPearl.EqualConstant(d[i,j], 0, trailer)) end
    end
    for i = 1:n, j=1:m
        if (j<m) push!(model.constraints, SeaPearl.Equal(r[i,j], l[i,j+1], trailer)) end
        if (i<n) push!(model.constraints, SeaPearl.Equal(d[i,j], u[i+1,j], trailer)) end
    end
    push!(model.constraints, SeaPearl.AllDifferent(id, trailer))
    model.adhocInfo = Dict([("n", n), ("m", m)])
    return model
end

"""
    solve_eternity2(input_file; variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Solve the SeaPearl model for to the eternity2 problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent and SeaPearl.TableConstraint, and the function model_AIS

# Arguments
- `n::Int`: dimension
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function solve_eternity2(input_file; order=[1,2,3,4], variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    model = model_eternity2(input_file; order=order,variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

"""
    solve_eternity2(model::SeaPearl.CPModel;benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Solve the SeaPearl model for to the eternity2 problem, using an existing model (from model_eternity2)

# Arguments
- `model::SeaPearl.CPModel`: model (from model_AIS)
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()

"""
function solve_eternity2(model::SeaPearl.CPModel; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic(),out_solver=false)
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection, out_solver=out_solver)
    return model
end

"""
    outputFromSeaPearl(model::SeaPearl.CPModel; optimality=false)

Return the results of the eternity2 problem as type OutputDataEternity, taking the solved model as argument

# Arguments
- `model::SeaPearl.CPModel`: needs the model to be already solved (by solve_eternity2)
"""
function outputFromSeaPearl(model::SeaPearl.CPModel; optimality=false)
    solutions = model.solutions
    nb_sols = length(solutions)
    n = model.adhocInfo["n"]
    m = model.adhocInfo["m"]
    orientation = Array{Int, 4}(undef, nb_sols, n,m,5)
    for (ind,sol) in enumerate(solutions)
        for i in 1:n, j in 1:m
            orientation[ind, i, j,1]= sol["id_"*string(i)*string(j)]
            orientation[ind, i, j, 2] = sol["u_"*string(i)*string(j)]
            orientation[ind, i, j, 3] = sol["r_"*string(i)*string(j)]
            orientation[ind, i, j, 4] = sol["d_"*string(i)*string(j)]
            orientation[ind, i, j, 5] = sol["l_"*string(i)*string(j)]
        end
    end
    return OutputDataEternityII(nb_sols, orientation)
end
