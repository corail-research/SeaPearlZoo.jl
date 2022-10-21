using SeaPearl
include("jobshopIOmanager.jl")


"""
solve_jobshop(filename::String)

Return the SeaPearl model solved for to the Job-shop scheduling problem, using SeaPearl.MinDomainVariableSelection and SeaPearl.BasicHeuristic(myHeuristic)

# Arguments
- `filename`: file containing the job times and job order

# Constraints
- Disjunctive
- isLessOrEqual

# Objective: minimize the makespan (i.e. the biggest end time)

# Branchable variables
- job_start (n x m size matrix) where: 
    n is the number of jobs
    m is the number of machines
    job_start[i, j] = 33 => job i starts on machine j at time = 33
"""
function solve_jobshop(filename::String)
    # TODO fix problem with Disjunctive constraint (change IntVar for AbstractIntVar)

    ### Initialization ###
    inputData = getJobshopInputData(filename)
    numberOfMachines = inputData.numberOfMachines   
    numberOfJobs = inputData.numberOfJobs       
    maxTime = inputData.maxTime            
    job_times = inputData.job_times          
    job_order = inputData.job_order   

    trailer = SeaPearl.Trailer()
    model = SeaPearl.CPModel(trailer)
    model.adhocInfo = Dict("numberOfMachines" => numberOfMachines, "numberOfJobs" => numberOfJobs)

    ### Variable declaration ###    

    # Start/End times for each machine and jobs
    job_start = Matrix{SeaPearl.AbstractIntVar}(undef, numberOfJobs, numberOfMachines)
    job_end = Matrix{SeaPearl.AbstractIntVar}(undef, numberOfJobs, numberOfMachines)

    # add variables
    for i in 1:numberOfJobs
        for j in 1:numberOfMachines
            job_start[i,j] = SeaPearl.IntVar(0, maxTime, "job_start_"*string(i)*"_"*string(j), model.trailer)
            SeaPearl.addVariable!(model, job_start[i,j]; branchable=true)
            job_end[i,j] = SeaPearl.IntVarViewOffset(job_start[i,j], job_times[i,j], "job_end_"*string(i)*"_"*string(j))
            SeaPearl.addVariable!(model, job_end[i,j]; branchable=false)
            # Set maxTime as upper bound to each job_end
            SeaPearl.addConstraint!(model, SeaPearl.LessOrEqualConstant(job_end[i,j] , maxTime, trailer))
        end
    end

    ### Constraints ###

    # ensure non-overlaps of the jobs
    for i in 1:numberOfJobs
        SeaPearl.addConstraint!(model, SeaPearl.Disjunctive(job_start[i, :], job_times[i,:], trailer))
    end

    # ensure non-overlaps of the machines
    for i in 1:numberOfMachines
        SeaPearl.addConstraint!(model, SeaPearl.Disjunctive(job_start[:, i], job_times[:, i], trailer))
    end

    # ensure the job order
    for job in 1:numberOfJobs
        for machine1 in 1:numberOfMachines
            for machine2 in 1:numberOfMachines
                if machine1 < machine2
                    if job_order[job,machine1] < job_order[job,machine2]
                        SeaPearl.addConstraint!(model, SeaPearl.LessOrEqual(job_end[job, machine1] , job_start[job, machine2], trailer))
                    else
                        SeaPearl.addConstraint!(model, SeaPearl.LessOrEqual(job_end[job, machine2] , job_start[job, machine1], trailer))
                    end
                end
            end
        end
    end

    ### Objective ###

    # TODO? use (n*m - 1) binaryMax constraints to get max value of job_ends 

    ### Solve ###
    model.limit.numberOfSolutions = 1
    model.limit.searchingTime = 60
    my_heuristic(x::SeaPearl.IntVar; cpmodel=nothing) = minimum(x.domain)
    SeaPearl.solve!(model; variableHeuristic=SeaPearl.MinDomainVariableSelection{true}(), valueSelection=SeaPearl.BasicHeuristic(my_heuristic))
    return model

end

# solved_model = solve_jobshop("./data/js_3_3")
# print_solution(solved_model)