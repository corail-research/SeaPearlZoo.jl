mutable struct Solution
    content     :: AbstractArray{Bool}
    value       :: Int
    weight      :: Int
    optimality  :: Bool
end

struct Item
    id      :: Int
    value   :: Int
    weight  :: Int
end

struct InputData
    items               :: AbstractArray{Union{Item, Nothing}}
    sortedItems         :: AbstractArray{Union{Item, Nothing}}
    numberOfItems       :: Int
    capacity            :: Int
end

function printSolution(solution::Solution)
    println(solution.value, " ", solution.optimality ? 1 : 0)
    for taken in solution.content
        print(taken ? 1 : 0, " ")
    end
    println()
end

function parseFile!(filename)
    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end
    data = nothing
    open(filename, "r") do openedFile
        data = parseInput!(read(openedFile, String))
    end
    
    return data
end

function parseInput!(input_data)
    lines = split(input_data, '\n')
    firstLine = split(lines[1], ' ')
    numberOfItems = parse(Int, firstLine[1])
    capacity = parse(Int, firstLine[2])
    items = Array{Union{Item}}(undef, numberOfItems);

    @assert numberOfItems + 2 <= length(lines)

    for i in 1:numberOfItems
        itemArray = split(lines[i+1], ' ')
        item = Item(i, parse(Int, itemArray[1]), parse(Int, itemArray[2]))
        items[i] = item
    end

    return InputData(items, Item[], numberOfItems, capacity)
end