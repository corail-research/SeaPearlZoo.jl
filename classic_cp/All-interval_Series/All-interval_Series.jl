using SeaPearl

struct OutputDataAIS
    nb_sols ::Int
    indices ::Matrix{Int}
    diffs ::Matrix{Int}
end

"""
    model_AIS(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

return the SeaPearl model for to the All-interval_Series (AIS) problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent  and SeaPearl.Asolute (without solving it)

# Arguments
- `n::Int`: dimension
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function model_AIS(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    series = Vector{SeaPearl.AbstractIntVar}(undef, n)
    series[1] = SeaPearl.IntVar(0, n-1, "s_"*string(1), trailer)
    SeaPearl.addVariable!(model, series[1]; branchable=true)

    interval_vector = Vector{SeaPearl.AbstractIntVar}(undef, n-1)
    interval_vector2 = Vector{SeaPearl.AbstractIntVar}(undef, n-1)

    for i = 1:n-1
        series[i+1] = SeaPearl.IntVar(0, n-1, "s_"*string(i+1), trailer)
        SeaPearl.addVariable!(model, series[i+1]; branchable=true)

        interval_vector[i] = SeaPearl.IntVar(-n+1, n-1, "v1_"*string(i), trailer)
        #SeaPearl.addVariable!(model, interval_vector[i]; branchable=false)

        minus = SeaPearl.IntVarViewOpposite(series[i+1], "-s["*string(i+1)*"]")
        vars = SeaPearl.AbstractIntVar[interval_vector[i], minus, series[i]]
        transition = SeaPearl.SumToZero(vars, trailer)
        push!(model.constraints, transition)

        interval_vector2[i] = SeaPearl.IntVar(1, n-1, "v2_"*string(i), trailer)
        #SeaPearl.addVariable!(model, interval_vector2[i]; branchable=false)
        abs_constraint = SeaPearl.Absolute(interval_vector[i], interval_vector2[i], trailer)
        push!(model.constraints, abs_constraint)
    end


    push!(model.constraints, SeaPearl.AllDifferent(series, trailer))
    push!(model.constraints, SeaPearl.AllDifferent(interval_vector2, trailer))

    return model
end

"""
    solve_AIS(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Solve the SeaPearl model for to the All-interval_Series (AIS) problem, using SeaPearl.MinDomainVariableSelection heuristique
and  SeaPearl.AllDifferent and SeaPearl.Absolute, and the function model_AIS

# Arguments
- `n::Int`: dimension
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function solve_AIS(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    model = model_AIS(n; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

"""
    solve_AIS(model::SeaPearl.CPModel)

Solve the SeaPearl model for to the AIS problem, using an existing model (from model_AIS)

# Arguments
- `model::SeaPearl.CPModel`: model (from model_AIS)

"""
function solve_AIS(model::SeaPearl.CPModel)
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    return model
end

"""
    outputFromSeaPearl(model::SeaPearl.CPModel; optimality=false)

Return the results of the AIS problem as type OutputDataAIS, taking the solved model (from solve_AIS) as argument

# Arguments
- `model::SeaPearl.CPModel`: needs the model to be already solved (by solve_AIS)
"""
function outputFromSeaPearl(model::SeaPearl.CPModel; optimality=false)
    solutions = model.solutions
    nb_sols = length(solutions)
    n = length(model.variables)
    indices = Matrix{Int}(undef, n, nb_sols)
    diffs = Matrix{Int}(undef, n-1, nb_sols)
    for (ind,sol) in enumerate(solutions)
        for i in 1:n
            indices[i, ind]= sol["s_"*string(i)]
            if (i<n) diffs[i,ind] = sol["s_"*string(i+1)]- sol["s_"*string(i)] end
        end
    end
    return OutputDataAIS(nb_sols, indices, diffs)
end

"""
outputFromSeaPearl(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

Shows the results of the AIS problem as type OutputDataAIS, taking dimension as argument

# Arguments
- `n::Int`: dimension
- 'variableSelection': SeaPearl variable selection. By default: SeaPearl.MinDomainVariableSelection{false}()
- 'valueSelection': SeaPearl value selection. By default: =SeaPearl.BasicHeuristic()
"""
function outputFromSeaPearl(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    model = solve_AIS(n::Int; benchmark=false, variableSelection=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    return outputFromSeaPearl(model)    
end
