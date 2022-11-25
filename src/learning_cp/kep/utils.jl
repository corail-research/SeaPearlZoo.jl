using BSON: @save
using Dates
using Flux
using JSON
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl

include("kep_config.jl")

struct select_random_value_kep <: Function
    rng::MersenneTwister
    function select_random_value_kep(rng::MersenneTwister)
        return new(rng)
    end
end

function (func::select_random_value_kep)(x::SeaPearl.IntVar; cpmodel=nothing)
    selected_number = rand(func.rng, 1: length(x.domain))
    i = 1
    for value in x.domain
        if i == selected_number
            return value
        end
        i += 1
    end
    @assert false "This should not happen"
end

function get_random_generator_from_seed(seed::Union{Nothing,Int}):: MersenneTwister
    if !isnothing(seed)
        random_generator = MersenneTwister(seed)
    else
        random_generator = MersenneTwister()
    end 
    return random_generator
end

function save_experiment_config(kep_parameters::KepParameters):: String
    experiment_start_time = now()
    rounded_experiment_start_time = round(experiment_start_time, Dates.Second(3))
    experiment_directory_name = string("exp_", Base.replace("$(rounded_experiment_start_time)",":"=>"-"))
    directory = mkdir(experiment_directory_name)
    open(dir*"/params.json", "w") do file
        JSON.print(file, Dict(fn=>string(getfield(args, fn)) for fn ∈ fieldnames(kep_parameters)))
    end
    return directory
end

function save_experiment_weights(agent::RL.Agent, directory::String):: Nothing
    trainedWeights = Flux.params(agent.policy.learner.approximator.model)
    @save directory*"/model_weights_kep"*string(args.num_nodes)*".bson" trainedWeights
end