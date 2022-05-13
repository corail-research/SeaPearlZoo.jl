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

# -------------------
# Generator
# -------------------

n_city = 10
grid_size = 25
max_tw_gap = 0
max_tw = 100
tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)


SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
SR2 = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Internal variables
# -------------------
function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, TS}}) where TS
    return 3
end
numInFeatures = SeaPearl.feature_length(SR)
numInFeatures2 = [1, 16, 1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 2001
evalFreq = 200
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
include("nearest_heuristic.jl")
nearest_heuristic = SeaPearl.BasicHeuristic(select_nearest_neighbor) # Basic value-selection heuristic

valueSelectionArray = [learnedHeuristic, learnedHeuristic2, nearest_heuristic]

# -------------------
# Variable Heuristic definition
# -------------------
struct TsptwVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
TsptwVariableSelection(;take_objective=false) = TsptwVariableSelection{take_objective}()
function (::TsptwVariableSelection{false})(cpmodel::SeaPearl.CPModel; rng=nothing)
    for i in 1:length(keys(cpmodel.variables))
        if haskey(cpmodel.variables, "a_"*string(i)) && !SeaPearl.isbound(cpmodel.variables["a_"*string(i)])
            return cpmodel.variables["a_"*string(i)]
        end
    end
end
variableSelection = TsptwVariableSelection()

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
        generator=tsptw_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose=true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray, tsptw_generator; evalFreq=evalFreq, nbInstances=nbInstances),
        restartPerInstances=1
    )

    #saving model weights
    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir * "/model_weights_tsptw_$(n_city).bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir * "/tsptw_$(n_city)_training")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir * "/tsptw_$(n_city)_training2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 1]; filename=dir * "/tsptw_$(n_city)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:, 2]; filename=dir * "/tsptw_$(n_city)_trained2")
    SeaPearlExtras.storedata(eval_metricsArray[:, 3]; filename=dir * "/tsptw_$(n_city)_min")

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
