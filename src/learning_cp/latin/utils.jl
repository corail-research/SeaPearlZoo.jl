using BSON: @save
using Dates
using Flux
using JSON
using SeaPearl

include("latin_config.jl")


function save_experiment_config(latin_exp_parameters::LatinExperimentConfig):: String
    experiment_start_time = now()
    rounded_experiment_start_time = round(experiment_start_time, Dates.Second(3))
    experiment_directory_name = string("exp_", Base.replace("$(rounded_experiment_start_time)",":"=>"-"))
    directory = mkdir(experiment_directory_name)
    open(dir*"/params.json", "w") do file
        JSON.print(file, Dict(fn=>string(getfield(args, fn)) for fn ∈ fieldnames(latin_exp_parameters)))
    end
    return directory
end

function save_experiment_weights(agent::RL.Agent, directory::String):: Nothing
    trainedWeights = Flux.params(agent.policy.learner.approximator.model)
    @save directory*"/model_weights_kep"*string(args.num_nodes)*".bson" trainedWeights
end

function select_random_value(x::SeaPearl.IntVar; cpmodel=nothing)
    selected_number = rand(1:length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
            return value
        end
        i += 1
    end
    @assert false "This should not happen"
end