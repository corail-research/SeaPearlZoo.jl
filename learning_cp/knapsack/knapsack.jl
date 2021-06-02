using Revise
using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using CUDA
using DataFrames
using CSV
using Statistics
using Zygote
using GeometricFlux
using Random
using BSON: @save, @load

using Plots
gr()

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
knapsack_generator = SeaPearl.KnapsackGenerator(5, 10, 0.2)

# -------------------
# Internal variables
# -------------------
const StateRepresentation = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(knapsack_generator, StateRepresentation)

# -------------------
# Experience variables
# -------------------
nbEpisodes = 30
evalFreq = 10
nbInstances = 3
nbRandomHeuristics = 0

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{StateRepresentation, knapsackReward, SeaPearl.FixedOutput}(agent)
basicHeuristic = SeaPearl.BasicHeuristic((x; cpmodel=nothing) -> SeaPearl.maximum(x.domain)) # Basic value-selection heuristic

# -------------------
# Variable Heuristic definition
# ------------------- 
struct KnapsackVariableSelection <: SeaPearl.AbstractVariableSelection{false} end

function (::KnapsackVariableSelection)(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables["x[" * string(i) * "]"])
        i += 1
    end
    return model.variables["x[" * string(i) * "]"]
end

valueSelectionArray = [learnedHeuristic, basicHeuristic]

# -------------------
# -------------------
# Core function
# -------------------
# -------------------
function trytrain(nbEpisodes::Int)

    metricsArray, eval_metricsArray = SeaPearl.train!(;
        valueSelectionArray= valueSelectionArray, 
        generator=knapsack_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch,
        variableHeuristic=KnapsackVariableSelection(),
        out_solver=false,
        verbose=true, #true to print processus
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,knapsack_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        metrics=nothing
        )

    #saving model weights
    trained_weights = params(approximator_model)
    @save "model_weights_knapsack"*string(knapsack_generator.nb_items)*".bson" trained_weights
    
    return metricsArray, eval_metricsArray
end

CUDA.allowscalar(false)
metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing


