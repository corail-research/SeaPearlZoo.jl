using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using BSON: @save, @load
using Random
using Statistics

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
nqueens_generator = SeaPearl.NQueensGenerator(15)
#model = model_queens(4)
SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
#gplot(SR.cplayergraph)

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(SR)

# -------------------
# Experience variables
# -------------------
nbEpisodes = 500
evalFreq = 100
nbInstances = 20
nbRandomHeuristics = 0

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SR, InspectReward, SeaPearl.FixedOutput}(agent)

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


    metricsArray, eval_metricsArray = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=nqueens_generator,
        nbEpisodes=nbEpisodes,
        strategy=SeaPearl.DFSearch(),
        variableHeuristic=variableSelection,
        out_solver=false,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,nqueens_generator; evalFreq = evalFreq, nbInstances = nbInstances),
        restartPerInstances = 1
    )

    #saving model weights

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
