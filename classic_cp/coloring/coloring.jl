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

    for key in keys(sol)
        color = sol[key]
        if !(color in edgeColors)
            numberOfColors += 1
        end
        push!(edgeColors, color)
    end
    return OutputData(numberOfColors, edgeColors, optimality)
end

function solve_coloring(input_file; benchmark=false)
    input = getInputData(input_file)
    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)

    ### Variable declaration ###
    x = SeaPearl.IntVar[]
    for i in 1:input.numberOfVertices
        push!(x, SeaPearl.IntVar(1, input.numberOfVertices, string(i), trailer))
        SeaPearl.addVariable!(model, last(x))
    end

    ### Constraints ###
    # Breaking some symmetries
    push!(model.constraints, SeaPearl.EqualConstant(x[1], 1, trailer))
    push!(model.constraints, SeaPearl.LessOrEqual(x[1], x[2], trailer))

    # Edge constraints
    degrees = zeros(Int, input.numberOfVertices)
    for e in input.edges
        push!(model.constraints, SeaPearl.NotEqual(x[e.vertex1], x[e.vertex2], trailer))
        degrees[e.vertex1] += 1
        degrees[e.vertex2] += 1
    end
    sortedPermutation = sortperm(degrees; rev=true)

    ### Objective ###
    numberOfColors = SeaPearl.IntVar(0, input.numberOfVertices, "numberOfColors", trailer)
    SeaPearl.addVariable!(model, numberOfColors)
    for var in x
        push!(model.constraints, SeaPearl.LessOrEqual(var, numberOfColors, trailer))
    end
    SeaPearl.addObjective!(model, numberOfColors)

    # TODO fix function selectVariable in order to use it for solving the problem
    ### Variable selection heurstic ###
    # function selectVariable(model::SeaPearl.CPModel, sortedPermutation, degrees)
    #     maxDegree = 0
    #     toReturn = nothing
    #     for i in sortedPermutation
    #         if !SeaPearl.isbound(model.variables[string(i)])
    #             if isnothing(toReturn)
    #                 toReturn = model.variables[string(i)]
    #                 maxDegree = degrees[i]
    #             end
    #             if degrees[i] < maxDegree
    #                 return toReturn
    #             end

    #             if length(model.variables[string(i)].domain) < length(toReturn.domain)
    #                 toReturn = model.variables[string(i)]
    #             end
    #         end
    #     end
    #     return toReturn
    # end

    # return model

    # TODO: When selectVariable() will work, remove the current call to solve!() to use the commented one
    SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())
    # status = SeaPearl.solve!(model; variableHeuristic=((m; cpmodel=nothing) -> selectVariable(m, sortedPermutation, degrees)))

    if !benchmark
        for oneSolution in model.statistics.solutions
            if !isnothing(oneSolution)
                output = outputFromSeaPearl(oneSolution)
                printSolution(output)
            end
        end
    end
end

# TODO remove those 3 commented lines if packages are not needed
# using Gadfly
# using LightGraphs
# using GraphPlot

# TODO remove function testGraph() if not usefull
# function testGraph(input_file)
#     model = solve_coloring(input_file)
#     g = SeaPearl.CPLayerGraph(model)
#     nodelabel = map((x) -> SeaPearl.labelOfVertex(g, x)[1], collect(1:nv(g)))

#     nlist = Vector{Int}[] # two shells
#     push!(nlist, collect((g.numberOfConstraints+1):(g.numberOfConstraints+g.numberOfVariables))) # second shell
#     push!(nlist, collect((g.numberOfConstraints+g.numberOfVariables+1):(g.numberOfConstraints+g.numberOfVariables+g.numberOfValues))) # second shell
#     push!(nlist, collect(1:g.numberOfConstraints)) # first shell

#     membership = map((x) -> SeaPearl.labelOfVertex(g, x)[2], collect(1:nv(g)))
#     nodecolor = [colorant"red", colorant"purple", colorant"green"]
#     # membership color
#     nodefillc = nodecolor[membership]
#     # locs_x, locs_y = shell_layout(g, nlist)
#     return g, nodefillc, nodelabel
# end
