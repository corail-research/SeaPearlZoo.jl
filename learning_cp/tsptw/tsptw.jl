using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Zygote
using GeometricFlux
using Statistics
using Random
using BSON: @load, @save
using DataFrames
using CSV
using Plots
gr()

# -------------------
# Generator
# -------------------
n_city = 10
grid_size = 100
max_tw_gap = 10
max_tw = 100
tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)

# -------------------
# Internal variables
# -------------------
numInFeatures=SeaPearl.feature_length(tsptw_generator,SeaPearl.TsptwStateRepresentation{SeaPearl.TsptwFeaturization})
state_size = SeaPearl.arraybuffer_dims(tsptw_generator, SeaPearl.TsptwStateRepresentation{SeaPearl.TsptwFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nbEpisodes = 4000
evalFreq = 200
nbInstances = 1
nbRandomHeuristics = 1

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.TsptwStateRepresentation, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)
include("nearest_heuristic.jl")
nearest_heuristic = SeaPearl.BasicHeuristic(select_nearest_neighbor) # Basic value-selection heuristic

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

valueSelectionArray = [learnedHeuristic, nearest_heuristic]
append!(valueSelectionArray, randomHeuristics)

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

    metricsArray, eval_metricsArray = SeaPearl.train!(
    valueSelectionArray=valueSelectionArray,
    generator=tsptw_generator,
    nbEpisodes=nbEpisodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=false,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,tsptw_generator; evalFreq = evalFreq, nbInstances = nbInstances)
)

    trained_weights = params(approximator_model)
    @save "model_weights_tsptw"*string(n_city)*".bson" trained_weights

    return metricsArray, eval_metricsArray
end

metricsArray, eval_metricsArray = trytrain(nbEpisodes)
SeaPearl.plotNodeVisited(metricsArray[1])
