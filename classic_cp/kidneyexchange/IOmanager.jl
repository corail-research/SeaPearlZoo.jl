function parseInput(raw_input)
    lines = split(raw_input, '\n')
    firstLine = split(lines[1], ' ')

    numberOfPairs = parse(Int, firstLine[1])
    density = parse(Float16, firstLine[2])

    compabilities = Vector{Union{Vector{Int}, Nothing}}(undef, numberOfPairs)
    fill!(compabilities, nothing)

    for i = 2:numberOfPairs+1
        if isempty(lines[i])
            compabilities[i-1] = []
        else
            line = split(lines[i],' ')
            compabilities[i-1] = map(x->parse(Int,x), line)
        end
    end
    return InputData(numberOfPairs, density, compabilities)
end

function getInputData(filename)
    inputData = nothing

    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end

    open(filename, "r") do openedFile
        inputData = parseInput(read(openedFile, String))
    end
    return inputData
end