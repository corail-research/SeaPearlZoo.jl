using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using BSON: @save, @load
using JSON
using Random
using Dates
using Statistics
using LightGraphs

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
board_size = 15
nqueens_generator = SeaPearl.NQueensGenerator(board_size)


# -------------------
# Features
# -------------------
features_type = BetterFeaturization

SR = SeaPearl.DefaultStateRepresentation{features_type, SeaPearl.DefaultTrajectoryState}

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(SR)

# -------------------
# Experience variables
# -------------------
nbEpisodes = 10000
evalFreq = 1000
nbInstances = 50
nbRandomHeuristics = 0
restartPerInstances = 1

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
rewardType = SeaPearl.GeneralReward
learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR,rewardType,SeaPearl.FixedOutput}(agent)
# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

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

randomHeuristics = []
for i in 1:nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learnedHeuristic, heuristic_min]
append!(valueSelectionArray, randomHeuristics)
# -------------------
# Variable Heuristic definition
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

function trytrain(nbEpisodes::Int)
    experienceTime = now()
    dir = mkdir(string("exp_", Base.replace("$(round(experienceTime, Dates.Second(3)))", ":" => "-")))
    out_solver = true
    expParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => nbEpisodes,
            :restartPerInstances => restartPerInstances,
            :evalFreq => evalFreq,
            :nbInstances => nbInstances,
        ),
        :generatorParameters => Dict(
            :boardSize => board_size,
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => features_type,
            :chosen_features => nothing
        ),
        :learnerParameters => Dict(
            :model => string(agent.policy.learner.approximator.model),
            :gamma => agent.policy.learner.sampler.γ,
            :batch_size => agent.policy.learner.sampler.batch_size,
            :update_horizon => agent.policy.learner.sampler.n,
            :min_replay_history => agent.policy.learner.min_replay_history,
            :update_freq => agent.policy.learner.update_freq,
            :target_update_freq => agent.policy.learner.target_update_freq,
        ),
        :explorerParameters => Dict(
            :ϵ_stable => agent.policy.explorer.ϵ_stable,
            :decay_steps => agent.policy.explorer.decay_steps,
        ),
        :trajectoryParameters => Dict(
            :trajectoryType => typeof(agent.trajectory),
            :capacity => trajectory_capacity
        ),
        :reward => rewardType
    )
    open(dir * "/params.json", "w") do file
        JSON.print(file, expParameters)
    end
    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=nqueens_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=out_solver,
        verbose=false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, nqueens_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_nqueens_$(board_size).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/nqueens_$(board_size)_training")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/nqueens_$(board_size)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/nqueens_$(board_size)_min")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:, i+2]; filename=dir * "/nqueens_$(board_size)_random$(i)")
    end

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
