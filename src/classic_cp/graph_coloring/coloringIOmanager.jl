struct Edge
    vertex1     :: Int
    vertex2     :: Int
end

struct ColoringInputData
    edges               :: Array{Edge}
    numberOfEdges       :: Int
    numberOfVertices    :: Int
end

struct OutputData
    numberOfColors      :: Int
    edgeColors          :: Array{Int}
    optimality          :: Bool
end

function parseColoringInput(raw_input)
    lines = split(raw_input, '\n')

    firstLine = split(lines[1], ' ')

    numberOfVertices = parse(Int, firstLine[1])
    numberOfEdges = parse(Int, firstLine[2])

    edges = Array{Union{Edge, Nothing}}(nothing, numberOfEdges)
    

    @assert numberOfEdges + 2 <= length(lines)

    for i in 1:numberOfEdges
        edgeArray = split(lines[i+1], ' ')
        edge = Edge(parse(Int, edgeArray[1])+1, parse(Int, edgeArray[2])+1)
        edges[i] = edge
    end

    return ColoringInputData(edges, numberOfEdges, numberOfVertices)
end

function solutionString(solution::OutputData)
    toReturn = ""
    toReturn *= string(solution.numberOfColors) * " " * string(solution.optimality ? 1 : 0) * "\n"
    for color in solution.edgeColors
        toReturn *= string(color) * " "
    end
    toReturn *= "\n"
end

function printSolution(solution::OutputData)
    println(solution.numberOfColors, " ", solution.optimality ? 1 : 0)
    for color in solution.edgeColors
        print(color, " ")
    end
    println()
end


function writeSolution(solution::OutputData, filename::String)
    open(filename, "w") do openedFile
        write(openedFile, solutionString(solution))
    end
    return
end

function getColoringInputData(filename)
    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end

    input = ColoringInputData([], 0, 0)

    open(filename, "r") do openedFile
        input = parseColoringInput(read(openedFile, String))
    end

    return input
end
