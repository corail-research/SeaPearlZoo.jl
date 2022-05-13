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

nbNodes = 50
nbMinColor = 10
density = 0.95

coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(nbNodes, nbMinColor, density)
SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
SR2 = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Internal variables
# -------------------
function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, TS}}) where TS
    return 3
end
numInFeatures = SeaPearl.feature_length(SR)
numInFeatures2 = [1, 2, 1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 10001
restartPerInstances = 1
evalFreq = 100
nbInstances = 1

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
learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, SeaPearl.GeneralReward, SeaPearl.FixedOutput}(agent)
learnedHeuristic2 = SeaPearl.SimpleLearnedHeuristic{SR2, SeaPearl.GeneralReward, SeaPearl.FixedOutput}(agent2; chosen_features=chosen_features)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

valueSelectionArray = [learnedHeuristic, learnedHeuristic2, heuristic_min]

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
        generator=coloring_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose=true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, coloring_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_coloring_$(nbNodes).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/coloring_$(nbNodes)_training")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir * "/coloring_$(nbNodes)_training2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/coloring_$(nbNodes)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/coloring_$(nbNodes)_trained2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 3]; filename=dir * "/coloring_$(nbNodes)_min")

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
