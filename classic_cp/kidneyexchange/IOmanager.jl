"""
parseInput(raw_input)

Return an InputData with the following fields:
- numberOfPairs::Int                 -> number of donor-patient pairs
- density::Float16                   -> probability of compatibility between 2 pairs (density ∈ [0,1])
- compabilities::Vector{Vector{Int}} -> v[i] contains the pairs that can give a kidney to pair i  

# Arguments
- `filename`: file containing the compatibilities between pairs

# Instance example: 
4 0.33 #numberOfPairs density
2 4    #pair 1 can receive a kidney from pairs 2 and 4
1      #pair 2 can receive a kidney from pair 1
4      #pair 3 can receive a kidney from pair 4
3      #pair 4 can receive a kidney from pair 3
"""
function parseInput(raw_input)
    lines = split(raw_input, '\n')
    firstLine = split(lines[1], ' ')

    numberOfPairs = parse(Int, firstLine[1])
    density = parse(Float16, firstLine[2])

    compabilities = Vector{Vector{Int}}(undef, numberOfPairs)

    for i = 2:numberOfPairs+1
        if isempty(lines[i])
            #Edge case: pair i can't receive any kidney
            compabilities[i-1] = []
        else
            line = split(lines[i],' ')
            compabilities[i-1] = map(x->parse(Int,x), line)
        end
    end
    return InputData(numberOfPairs, density, compabilities)
end

"""
getInputData(filename::String)

return the KEP instance corresponding to filemane

# Arguments
- `filename`: file containing the compatibilities between pairs
"""
function getInputData(filename::String)
    inputData = nothing
    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end

    open(filename, "r") do openedFile
        inputData = parseInput(read(openedFile, String))
    end
    return inputData
end