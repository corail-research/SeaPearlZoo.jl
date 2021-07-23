using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using Random

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
latin_generator = SeaPearl.LatinGenerator(11,0.6)
# -------------------
# Internal variables
# -------------------

SR = SeaPearl.DefaultStateRepresentation{LatinFeaturization, SeaPearl.DefaultTrajectoryState}
numInFeatures = SeaPearl.feature_length(SR)
numGlobalFeature = SeaPearl.global_feature_length(SR)
# -------------------
# Experience variables
# -------------------
nbEpisodes = 500
evalFreq = 30
nbInstances = 1
nbRandomHeuristics = 0

# -------------------
# Agent definition
# -------------------

include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic=SeaPearl.LearnedHeuristic{SR, InspectReward, SeaPearl.FixedOutput}(agent)
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

valueSelectionArray = [learnedHeuristic,heuristic_min]
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


    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=latin_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=false,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,latin_generator; evalFreq = evalFreq, nbInstances = nbInstances),
        restartPerInstances = 1
    )

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
