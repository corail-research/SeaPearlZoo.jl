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


#SR = SeaPearl.DefaultStateRepresentation{BetterFeaturization,SeaPearl.DefaultTrajectoryState}
# -------------------
# Features
# -------------------
constraint_activity = true
values_onehot = true
nb_possible_values = 15
variable_initial_domain_size = true
nb_involved_constraint_propagation = true

chosen_features = Dict([("constraint_activity", constraint_activity), ("values_onehot", values_onehot), ("variable_initial_domain_size", variable_initial_domain_size), ("nb_involved_constraint_propagation", nb_involved_constraint_propagation)])

# TODO: Edit it to automatically compute the number of constraint types
nb_features = 3
nb_constraint_types = 1
if values_onehot
    nb_features += nb_possible_values
else
    nb_features += 1
end
nb_features += constraint_activity + variable_initial_domain_size + nb_involved_constraint_propagation + nb_constraint_types

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.FeaturizationHelper, TS}}) where TS
    return nb_features
end


SR = SeaPearl.DefaultStateRepresentation{SeaPearl.FeaturizationHelper, SeaPearl.DefaultTrajectoryState}

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

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
#learnedHeuristic = SeaPearl.LearnedHeuristic{SR,SeaPearl.CPReward,SeaPearl.FixedOutput}(agent)
learnedHeuristic = SeaPearl.LearnedHeuristic{SR, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent; chosen_features = chosen_features)
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
        :nbEpisodes => nbEpisodes,
        :evalFreq => evalFreq,
        :nbInstances => nbInstances,
        :rlagent => string(agent),
        :out_solver => out_solver
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
