struct InputData
    n    ::Int
    m    ::Int
    pieces ::Matrix{Int}
end


function parseInput(raw_input; order=[1,2,3,4])
    lines = split(raw_input, '\n')

    firstLine = split(lines[1], ' ')

    n = parse(Int, firstLine[1])
    m = parse(Int, firstLine[2])

    pieces = Matrix{Int}(undef, n*m, 4)

    for i = 2:n*m+1
        line = split(lines[i],' ')
        for j =1:4
            pieces[i-1,j] = parse(Int, line[order[j]])
        end
    end

    return InputData(n, m, pieces)
end



function getInputData(filename;order=[1,2,3,4])

    inputData = nothing

    open(filename, "r") do openedFile
        inputData = parseInput(read(openedFile, String);order=order)
    end

    return inputData
end
