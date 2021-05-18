struct InputData
    n    ::Int
    m    ::Int
    pieces ::Matrix{Int}
end


function parseInput(raw_input)
    lines = split(raw_input, '\n')

    firstLine = split(lines[1], ' ')

    n = parse(Int, firstLine[1])
    m = parse(Int, firstLine[2])

    pieces = Matrix{Int}(undef, n*m, 4)

    for i = 2:n*m+1, j=1:4
        line = split(lines[i],' ')
        pieces[i-1,j] = parse(Int, line[j])
    end

    return InputData(n, m, pieces)
end



function getInputData(filename)

    inputData = nothing

    open(filename, "r") do openedFile
        inputData = parseInput(read(openedFile, String))
    end

    return inputData
end
