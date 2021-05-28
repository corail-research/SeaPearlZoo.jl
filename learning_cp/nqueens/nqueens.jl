using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Zygote
using GeometricFlux
using Random
using BSON: @save, @load
using DataFrames
using CSV
using Plots
using Statistics
gr()

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
nqueens_generator = SeaPearl.NQueensGenerator(8)
#model = model_queens(4)
#SR = SeaPearl.DefaultStateRepresentation{BetterFeaturization}(model)
#gplot(SR.cplayergraph)

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(nqueens_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
state_size = SeaPearl.arraybuffer_dims(nqueens_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 5000
evalFreq = 300
nbInstances = 1
nbRandomHeuristics = 1

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{BetterFeaturization}, InspectReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)

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
        strategy=SeaPearl.DFSearch,
        variableHeuristic=variableSelection,
        out_solver=false,
        verbose = true,
        evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,nqueens_generator; evalFreq = evalFreq, nbInstances = nbInstances)
    )

    #saving model weights
    trained_weights = params(approximator_model)
    @save "model_weights_gc"*string(nqueens_generator.board_size)*".bson" trained_weights

    return metricsArray, eval_metricsArray
end



# -------------------
# -------------------

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
