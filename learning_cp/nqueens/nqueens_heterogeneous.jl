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
board_size = 15
nqueens_generator = SeaPearl.NQueensGenerator(board_size)
SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
SR2 = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Internal variables
# -------------------
function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, TS}}) where TS
    return 3
end
numInFeatures = SeaPearl.feature_length(SR)
numInFeatures2 = [1, 5, 1]
println("numInFeatures ", SeaPearl.feature_length(SR))

# -------------------
# Experience variables
# -------------------
nbEpisodes = 10001
evalFreq = 500
nbInstances = 50
nbRandomHeuristics = 0

# -------------------
# Agent definition
# -------------------
include("agents_heterogeneous.jl")

# -------------------
# Value Heuristic definition
# -------------------

chosen_features = Dict(
    "constraint_activity" => false,
    "constraint_type" => true,
    "nb_involved_constraint_propagation" => false,
    "nb_not_bounded_variable" => false,
    "variable_domain_size" => false,
    "variable_initial_domain_size" => true,
    "variable_is_bound" => false,
    "values_onehot" => false,
    "values_raw" => true,
)

# learnedHeuristic = SeaPearl.LearnedHeuristic{SR, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)
learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent)
learnedHeuristic2 = SeaPearl.SimpleLearnedHeuristic{SR2, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent2; chosen_features=chosen_features)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

randomHeuristics = []
for i in 1:nbRandomHeuristics
    # push!(randomHeuristics, SeaPearl.RandomHeuristic())
end

valueSelectionArray = [learnedHeuristic, learnedHeuristic2, heuristic_min]
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
        verbose=true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, nqueens_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_nqueens_$(board_size).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/nqueens_$(board_size)_training")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir * "/nqueens_$(board_size)_training2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/nqueens_$(board_size)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/nqueens_$(board_size)_trained2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 3]; filename=dir * "/nqueens_$(board_size)_min")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:, i+2]; filename=dir * "/nqueens_$(board_size)_random$(i)")
    end

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
