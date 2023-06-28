using DataFrames
using CSV

function get_first_solution(solutions::Vector{Union{Nothing, Float32}})
    for i = 1:length(solutions)
        if !isnothing(solutions[i])
            return solutions[i]
        end
    end
    return nothing
end

function get_last_solution(solutions::Vector{Union{Nothing, Float32}})
    for i = length(solutions):-1:1
        if !isnothing(solutions[i])
            return solutions[i]
        end
    end
    return nothing
end


function save_metrics(eval_metricsArray::Matrix{SeaPearl.AbstractMetrics}, save_path::AbstractString)

    column_names = ["num_heuristic", "num_instance", "num_experiment", "heuristic_type", "reward_type", "policy_type", "first_sol", "last_sol", "node_visited_first_sol", "total_node_visited", "total_time"]
    # Check if the file exists
    if !isfile(save_path)
        CSV.write(save_path, DataFrame([]); header=column_names)
    end

    nb_instance, nb_heuristic = size(eval_metricsArray)

    nb_experiment = length(eval_metricsArray[1,1].scores)

    
    for j = 1:nb_heuristic
        num_heuristic = j
        heuristic = eval_metricsArray[1,j].heuristic
        heuristic_type = nothing
        policy_type = nothing
        reward_type = nothing
        if isa(heuristic, SeaPearl.SimpleLearnedHeuristic)
            if isa(heuristic.agent.policy, PPOPolicy)
                policy_type = "PPOPolicy"
            end
            if isa(heuristic.agent.policy, QBasedPolicy)
                policy_type = "QBasedPolicy"
            end
            reward_type = split(split(string(eval_metricsArray[1,j].heuristic.reward), "(")[1], ".")[2]
            heuristic_type = "SimpleLearnedHeuristic"
        end
        if isa(heuristic, SeaPearl.BasicHeuristic)
            heuristic_type = "BasicHeuristic(" * string(heuristic.selectValue) * ")"
        end
        for i = 1:nb_instance
            num_instance = i
            for k = 1:nb_experiment
                num_experiment = k
                first_sol = get_first_solution(eval_metricsArray[i,j].scores[k])
                last_sol = get_last_solution(eval_metricsArray[i,j].scores[k])
                node_visited_first_sol = eval_metricsArray[i,j].meanNodeVisitedUntilfirstSolFound[k]
                total_node_visited = eval_metricsArray[i,j].meanNodeVisitedUntilEnd[k]
                total_time = eval_metricsArray[i,j].TotalTimeNeeded[k]

                # Create the new row to add
                new_row = [num_heuristic, num_instance, num_experiment, heuristic_type, reward_type, policy_type, first_sol, last_sol, node_visited_first_sol, total_node_visited, total_time]
                new_row = map(x -> x === nothing ? "nothing" : x, new_row)

                # Convert the matrix to a DataFrame and Write the updated data to the CSV file
                CSV.write(save_path, DataFrame(hcat(new_row...), :auto); append=true)
            end
        end
    end
end