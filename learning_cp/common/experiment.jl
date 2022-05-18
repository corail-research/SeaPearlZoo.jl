using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using JSON
using BSON: @save, @load
using Dates
using Random
using LightGraphs
using OrderedCollections
using ArgParse

#####################
# Argument Parsing
#####################

s = ArgParseSettings()
@add_arg_table s begin
    "--episode", "-e"
    help = "number of episodes"
    arg_type = Int
    "--eval-freq", "-f"
    help = "number of episodes between each evaluation."
    arg_type = Int
    "--instance", "-i"
    help = "number of instances per evaluation."
    arg_type = Int
    "--restart"
    help = "number of restart per instances."
    arg_type = Int
    "--random", "-r"
    help = "number of random heuristic."
    arg_type = Int
    "--size", "-s"
    help = "size of the problem."
    arg_type = Int
    "--batch-size"
    help = "size of batches."
    arg_type = Int
    "--update-horizon"
    help = "update horizon."
    arg_type = Int
    "--min-replay-history"
    arg_type = Int
    "--update-freq"
    arg_type = Int
    "--target-update-freq"
    arg_type = Int
    "--decay-steps"
    arg_type = Int
    "--capacity"
    arg_type = Int
    "--conv-size"
    arg_type = Int
    "--dense-size"
    arg_type = Int
    "--verbose"
    help = "verbose output"
    action = :store_true
end

args = parse_args(s)
if !isnothing(args["episode"])
    NB_EPISODES = args["episode"]
end
if !isnothing(args["eval-freq"])
    EVAL_FREQ = args["eval-freq"]
end
if !isnothing(args["instance"])
    NB_INSTANCES = args["instance"]
end
if !isnothing(args["random"])
    NB_RANDOM_HEURISTICS = args["random"]
end
if !isnothing(args["restart"])
    RESTART_PER_INSTANCES = args["restart"]
end
if !isnothing(args["size"])
    SIZE = args["size"]
end
if !isnothing(args["batch-size"])
    BATCH_SIZE = args["batch-size"]
end
if !isnothing(args["update-horizon"])
    UPDATE_HORIZON = args["update-horizon"]
end
if !isnothing(args["min-replay-history"])
    MIN_REPLAY_HISTORY = args["min-replay-history"]
end
if !isnothing(args["update-freq"])
    UPDATE_FREQ = args["update-freq"]
end
if !isnothing(args["target-update-freq"])
    TARGET_UPDATE_FREQ = args["target-update-freq"]
end
if !isnothing(args["decay-steps"])
    DECAY_STEPS = args["decay-steps"]
end
if !isnothing(args["capacity"])
    CAPACITY = args["capacity"]
end
if !isnothing(args["conv-size"])
    CONV_SIZE = args["conv-size"]
end
if !isnothing(args["dense-size"])
    DENSE_SIZE = args["dense-size"]
end
if args["verbose"]
    VERBOSE = true
end

# -------------------
# -------------------
# Core function
# -------------------
# -------------------
using CircularArrayBuffers

function trytrain(; nbEpisodes::Int, evalFreq::Int, nbInstances::Int, restartPerInstances::Int, generator::SeaPearl.AbstractModelGenerator, variableHeuristic::SeaPearl.AbstractVariableSelection, learnedHeuristics::OrderedDict{String,<:SeaPearl.LearnedHeuristic}, basicHeuristics::OrderedDict{String,SeaPearl.BasicHeuristic}, expParameters=Dict{String,Any}()::Dict{String,Any}, base_name="experiment"::String, out_solver=true::Bool, verbose=false::Bool, nbRandomHeuristics=0::Int)
    experienceTime = now()
    dir = mkdir(string("exp_", Base.replace("$(round(experienceTime, Dates.Second(3)))", ":" => "-")))
    lh = last(collect(values(learnedHeuristics)))
    commonExpParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => nbEpisodes,
            :restartPerInstances => restartPerInstances,
            :evalFreq => evalFreq,
            :nbInstances => nbInstances,
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => typeof(lh).parameters[1].parameters[1],
            :chosen_features => lh.chosen_features
        ),
        :learnerParameters => Dict(
            :model => string(lh.agent.policy.learner.approximator.model),
            :gamma => lh.agent.policy.learner.sampler.γ,
            :batch_size => lh.agent.policy.learner.sampler.batch_size,
            :update_horizon => lh.agent.policy.learner.sampler.n,
            :min_replay_history => lh.agent.policy.learner.min_replay_history,
            :update_freq => lh.agent.policy.learner.update_freq,
            :target_update_freq => lh.agent.policy.learner.target_update_freq,
        ),
        :explorerParameters => Dict(
            :ϵ_stable => lh.agent.policy.explorer.ϵ_stable,
            :decay_steps => lh.agent.policy.explorer.decay_steps,
        ),
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(lh.agent.trajectory),
            :capacity => CircularArrayBuffers.capacity(lh.agent.trajectory.traces.action) - 1
        ),
        :reward => typeof(lh).parameters[2]
    )

    expParameters = merge(expParameters, commonExpParameters)
    open(dir * "/params.json", "w") do file
        JSON.print(file, expParameters)
    end

    randomHeuristics = Array{SeaPearl.BasicHeuristic}(undef, 0)
    for i in 1:nbRandomHeuristics
        push!(randomHeuristics, SeaPearl.RandomHeuristic())
    end

    valueSelectionArray = cat(collect(values(learnedHeuristics)), collect(values(basicHeuristics)), randomHeuristics, dims=1)

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableHeuristic,
        out_solver=out_solver,
        verbose=verbose,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=restartPerInstances
    )

    #saving model weights
    for (key, lh) in learnedHeuristics
        model = lh.agent.policy.learner.approximator
        @save dir * "/model_" * key * ".bson" model
    end

    counter = 0
    for key in keys(learnedHeuristics)
        counter += 1
        SeaPearlExtras.storedata(metricsArray[counter]; filename=dir * "/" * base_name * "_training_" * key)
    end

    counter = 0
    for key in keys(learnedHeuristics)
        counter += 1
        SeaPearlExtras.storedata(eval_metricsArray[:, counter]; filename=dir * "/" * base_name * "_" * key)
    end
    for key in keys(basicHeuristics)
        counter += 1
        SeaPearlExtras.storedata(eval_metricsArray[:, counter]; filename=dir * "/" * base_name * "_" * key)
    end
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:, counter+i]; filename=dir * "/" * base_name * "_random$(i)")
    end

    return metricsArray, eval_metricsArray
end
