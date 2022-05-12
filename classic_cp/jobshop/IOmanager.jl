function getInputData(filename::String)
    inputData = nothing
    if filename == ""
        throw(ArgumentError("You must specify a data file"))
    end

    content = open(f->read(f, String), filename)
    inputData = parseInput(content)

    return inputData
end

function parseInput(raw_input)
    lines = split(raw_input, '\n')
    firstLine = split(lines[1], ' ')

    numberOfMachines = parse(Int, firstLine[1])
    numberOfJobs = parse(Int, firstLine[2])
    maxTime = parse(Int, firstLine[3])

    job_times = Matrix{Int}(undef, numberOfJobs, numberOfMachines)
    job_order = Matrix{Int}(undef, numberOfJobs, numberOfMachines)

    for i in 1:numberOfJobs
        durations = split(lines[i+2], ' ') 
        orders = split(lines[i+3+numberOfJobs], ' ') 
        job_times[i,:] = map(x->parse(Int,x), durations)
        job_order[i,:] = map(x->parse(Int,x), orders)
    end

    return InputData(numberOfMachines, numberOfJobs, maxTime, job_times, job_order)
end