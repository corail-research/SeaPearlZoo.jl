using SeaPearl
include("coloringIOmanager.jl")


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
    input = getColoringInputData(input_file)
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

    ### Objective ###
    numberOfColors = SeaPearl.IntVar(0, input.numberOfVertices, "numberOfColors", trailer)
    SeaPearl.addVariable!(model, numberOfColors)
    for var in x
        push!(model.constraints, SeaPearl.LessOrEqual(var, numberOfColors, trailer))
    end
    SeaPearl.addObjective!(model, numberOfColors)

    SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{false}(), valueSelection=SeaPearl.BasicHeuristic())

    if !benchmark
        for oneSolution in model.statistics.solutions
            if !isnothing(oneSolution)
                output = outputFromSeaPearl(oneSolution)
                printSolution(output)
            end
        end
    end
end

# solve_coloring("./data/gc_4_1")