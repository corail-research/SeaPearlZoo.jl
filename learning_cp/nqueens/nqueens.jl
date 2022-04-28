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
include("rewards.jl")
include("features.jl")
# -------------------
# Generator
# -------------------
board_size = 5
nqueens_generator = SeaPearl.NQueensGenerator(board_size)
SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(SR)
# -------------------
# Experience variables
# -------------------
nbEpisodes = 2000
evalFreq = 100
nbInstances = 100
nbRandomHeuristics = 0
# -------------------
# Agent definition
# -------------------
include("agents.jl")
# -------------------
# Value Heuristic definition
# -------------------
eta = .9
learnedHeuristic = SeaPearl.SupervisedLearnedHeuristic{SR,SeaPearl.CPReward,SeaPearl.FixedOutput}(agent, eta)
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
    expParameters = Dict(
        :nbEpisodes => nbEpisodes,
        :evalFreq => evalFreq,
        :nbInstances => nbInstances
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
        out_solver=true,
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