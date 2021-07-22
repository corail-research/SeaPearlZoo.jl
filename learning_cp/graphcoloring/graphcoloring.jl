using SeaPearlExtras
using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Zygote
using GeometricFlux
using Random
using Dates
using Statistics


# -------------------
# Experience variables
# -------------------
nbEpisodes = 1000
restartPerInstances = 20
evalFreq = 50
nbInstances = 10
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
SR = SeaPearl.DefaultStateRepresentation{BetterFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(SR)
#numGlobalFeature = SeaPearl.global_feature_length(SR)

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------

learnedHeuristic=SeaPearl.LearnedHeuristic{SR, SeaPearl.DefaultReward2, SeaPearl.FixedOutput}(agent)
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
        :nbEpisodes => nbEpisodes,
        :restartPerInstances => restartPerInstances,
        :evalFreq => evalFreq,
        :nbInstances => nbInstances,
        :nbRandomHeuristics => nbRandomHeuristics,
        nbNodes => nbNodes,
        nbMinColor => nbMinColor,
        density => density
    )
    open(dir*"/params.json", "w") do file
        JSON.print(file, expParameters)
    end

    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=coloring_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.ILDSearch(0),
        variableHeuristic=variableSelection,
        out_solver=true,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,coloring_generator; evalFreq = evalFreq, nbInstances = nbInstances),
        restartPerInstances = restartPerInstances
    )

    #saving model weights
    trained_weights = params(approximator_model)
    @save dir*"model_weights_gc"*string(coloring_generator.n)*"_graph.bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/graph_coloring_$(nbNodes)_training_learned")
    SeaPearlExtras.storedata(metricsArray[2]; filename=dir*"/graph_coloring_$(nbNodes)_training_greedy")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(metricsArray[2+i]; filename=dir*"/graph_coloring_$(nbNodes)_training_random$(i)")
    end
    SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/graph_coloring_$(nbNodes)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/graph_coloring_$(nbNodes)_greedy")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:,i+2]; filename=dir*"/graph_coloring_$(nbNodes)_random$(i)")
    end

    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
