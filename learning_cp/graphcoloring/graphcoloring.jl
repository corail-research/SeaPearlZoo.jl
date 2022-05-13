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


# -------------------
# Experience variables
# -------------------
nbEpisodes = 1001
restartPerInstances = 1
evalFreq = 100
nbInstances = 50
nbRandomHeuristics = 1

nbNodes = 20
nbMinColor = 5
density = 0.95
# -------------------
# Generator
# -------------------
coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(nbNodes, nbMinColor, density)

#include("rewards.jl")
include("features.jl")

# -------------------
# Internal variables
# -------------------
featurizationType = BetterFeaturization
rewardType = SeaPearl.GeneralReward

SR = SeaPearl.DefaultStateRepresentation{featurizationType, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(SR)
#numGlobalFeature = SeaPearl.global_feature_length(SR)

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------

learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, rewardType, SeaPearl.FixedOutput}(agent)
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
    dir = mkdir(string("exp_",Base.replace("$(round(experienceTime, Dates.Second(3)))",":"=>"-")))
    expParameters = Dict(
        :experimentParameters => Dict(
            :nbEpisodes => nbEpisodes,
            :restartPerInstances => restartPerInstances,
            :evalFreq => evalFreq,
            :nbInstances => nbInstances,
        ),
        :generatorParameters => Dict(
            :nbNodes => nbNodes,
            :nbMinColor => nbMinColor,
            :density => density
        ),
        :nbRandomHeuristics => nbRandomHeuristics,
        :Featurization => Dict(
            :featurizationType => featurizationType,
            #:chosen_features => featurizationType == SeaPearl.FeaturizationHelper ? chosen_features : nothing
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
    open(dir*"/params.json", "w") do file
        JSON.print(file, expParameters)
    end

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=coloring_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,coloring_generator; evalFreq = evalFreq, nbInstances = nbInstances),
        restartPerInstances = restartPerInstances
    )

    #saving model weights
    model = agent.policy.learner.approximator
    @save dir*"/model_gc"*string(coloring_generator.n)*".bson" model

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/graph_coloring_$(nbNodes)_traininglearned")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir*"/graph_coloring_$(nbNodes)_traininggreedy")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(metricsArray[2+i]; filename=dir*"/graph_coloring_$(nbNodes)_trainingrandom$(i)")
    end
    SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/graph_coloring_$(nbNodes)_learned")
    SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/graph_coloring_$(nbNodes)_greedy")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:,i+2]; filename=dir*"/graph_coloring_$(nbNodes)_random$(i)")
    end

    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
