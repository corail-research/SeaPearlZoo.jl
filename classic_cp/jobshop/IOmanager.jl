struct InputData
    numberOfMachines   :: Int
    numberOfJobs       :: Int
    maxTime            :: Int
    job_times          :: Matrix{Int}
    job_order          :: Matrix{Int}
end

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

"""
print_solution(solved_model::SeaPearl.CPModel)

Print solutions found by `solve_jobshop()`

# Print format
- Job start
    job_start[i, j] = 33 => job i starts on machine j at time = 33
- Job end
    job_end[i, j] = 54 => job i ends on machine j at time = 54
"""
function print_solution(solved_model::SeaPearl.CPModel)
    solutions = solved_model.statistics.solutions
    realSolutions = filter(e -> !isnothing(e),solutions)
    numberOfMachines = solved_model.adhocInfo["numberOfMachines"]
    numberOfJobs = solved_model.adhocInfo["numberOfJobs"] 

    for solution in realSolutions
        println("Job start: ")
        for i in 1:numberOfJobs
            for j in 1:numberOfMachines
                id = "job_start_"*string(i)*"_"*string(j)
                val = string(solution[id])
                print(val*" ")
            end
            println()
        end
        println("Job end: ")
        for i in 1:numberOfJobs
            for j in 1:numberOfMachines
                id = "job_end_"*string(i)*"_"*string(j)
                val = string(solution[id])
                print(val*" ")
            end
            println()
        end
    end
end
