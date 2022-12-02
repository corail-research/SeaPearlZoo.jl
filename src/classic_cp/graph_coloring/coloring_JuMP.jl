using JuMP
using SeaPearl

struct Edge
    vertex1     :: Int
    vertex2     :: Int
end

struct InputData
    edges               :: Array{Edge}
    numberOfEdges       :: Int
    numberOfVertices    :: Int
end

struct OutputData
    numberOfColors      :: Int
    edgeColors          :: Array{Int}
    optimality          :: Bool
end

include("IOmanager.jl")


function outputFromSeaPearl(sol::SeaPearl.Solution; optimality=false)
    numberOfColors = 0
    edgeColors = Int[]

    for i in 1:length(keys(sol))
        color = sol[string(i)]
        if !(color in edgeColors)
            numberOfColors += 1
        end
        push!(edgeColors, color)
    end

    return OutputData(numberOfColors, edgeColors, optimality)
end

struct GraphColoringVariableSelection  <: SeaPearl.AbstractVariableSelection{true}
    sortedPermutation
    degrees
end
function (vs::GraphColoringVariableSelection)(model::SeaPearl.CPModel)
    sortedPermutation, degrees = vs.sortedPermutation, vs.degrees
    maxDegree = 0
    toReturn = nothing
    for i in sortedPermutation
        if !SeaPearl.isbound(model.variables[string(i)])
            if isnothing(toReturn)
                toReturn = model.variables[string(i)]
                maxDegree = degrees[i]
            end
            if degrees[i] < maxDegree
                return toReturn
            end

            if length(model.variables[string(i)].domain) < length(toReturn.domain)
                toReturn = model.variables[string(i)]
            end
        end
    end
    return toReturn
end

function solve_coloring_MOI(input_file; benchmark=false)

    # use input data to fill the model
    input = getInputData(input_file)

    model = SeaPearl.Optimizer()

    for i in 1:input.numberOfVertices
        MOI.add_constrained_variable(model, MOI.Interval(1, input.numberOfVertices))
    end

    degrees = zeros(Int, input.numberOfVertices)

    for e in input.edges
        MOI.add_constraint(model, MOI.VectorOfVariables([MOI.VariableIndex(e.vertex1), MOI.VariableIndex(e.vertex2)]), SeaPearl.VariablesEquality(false))
        degrees[e.vertex1] += 1
        degrees[e.vertex2] += 1
    end

    sortedPermutation = sortperm(degrees; rev=true)

    # define the heuristic used for variable selection
    variableheuristic = GraphColoringVariableSelection(sortedPermutation, degrees)
    MOI.set(model, SeaPearl.MOIVariableSelectionAttribute(), variableheuristic)

    MOI.set(model, SeaPearl.VariableSelection(), variableheuristic)

    solution = MOI.optimize!(model)

    output = outputFromSeaPearl(solution)
    printSolution(output)
end



function solve_coloring_JuMP(input_file; benchmark=false)

    # use input data to fill the model
    input = getInputData(input_file)

    model = Model(SeaPearl.Optimizer)

    @variable(model, 1 <= x[1:input.numberOfVertices] <= input.numberOfVertices)


    degrees = zeros(Int, input.numberOfVertices)
    for e in input.edges
        @constraint(model, [x[e.vertex1], x[e.vertex2]] in SeaPearl.NotEqualSet())
        #update degrees
        degrees[e.vertex1] += 1
        degrees[e.vertex2] += 1
    end

    @variable(model, 1 <= y <= input.numberOfVertices)
    for i in 1:input.numberOfVertices
        @constraint(model, x[i] <= y)
    end
    @objective(model, Min, y)

    sortedPermutation = sortperm(degrees; rev=true)

    # define the heuristic used for variable selection
    variableheuristic = GraphColoringVariableSelection(sortedPermutation, degrees)
    MOI.set(model, SeaPearl.MOIVariableSelectionAttribute(), variableheuristic)

    optimize!(model)
    status = MOI.get(model, MOI.TerminationStatus())

    # output = outputFromSeaPearl(solution)
    # printSolution(output)
    println(model)
    println(status)
    println(has_values(model))
    println(value.(x))
    println(value(y))

end
