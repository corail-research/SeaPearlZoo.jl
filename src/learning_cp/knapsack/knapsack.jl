using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using BSON: @save, @load
using Random
using Statistics
using Dates
using JSON


include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
knapsack_generator = SeaPearl.KnapsackGenerator(10, 10, 0.2)

# -------------------
# Internal variables
# -------------------
StateRepresentation = SeaPearl.DefaultStateRepresentation{KnapsackFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(StateRepresentation)

# -------------------
# Experience variables
# -------------------
nbEpisodes = 1000
evalFreq = 100
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
    experienceTime = now()
    dir = mkdir(string("exp_",Base.replace("$(round(experienceTime, Dates.Second(3)))",":"=>"-")))
    expParameters = Dict(
        :nbEpisodes => nbEpisodes,
        :evalFreq => evalFreq,
        :nbInstances => nbInstances
    )
    open(dir*"/params.json", "w") do file
        JSON.print(file, expParameters)
    end
    metricsArray, eval_metricsArray = SeaPearl.train!(;
        valueSelectionArray= valueSelectionArray,
        generator=knapsack_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=KnapsackVariableSelection(),
        out_solver=false,
        verbose=false, #true to print processus
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,knapsack_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances = 1
        )
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir*"/model_weights_knapsack.bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/knapsack_training")
    SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/knapsack_learned")
    SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/knapsack_basic")
    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
