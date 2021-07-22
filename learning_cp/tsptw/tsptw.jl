using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Zygote
using GeometricFlux
using Statistics
using Random
using BSON: @load, @save
using Dates

# -------------------
# Generator
# -------------------
n_city = 11
grid_size = 100
max_tw_gap = 0
max_tw = 1000
tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)
SR = SeaPearl.TsptwStateRepresentation{SeaPearl.TsptwFeaturization, SeaPearl.TsptwTrajectoryState}

# -------------------
# Internal variables
# -------------------
numInFeatures=SeaPearl.feature_length(SR)

# -------------------
# Experience variables
# -------------------
nbEpisodes = 2
evalFreq = 200
nbInstances = 10
nbRandomHeuristics = 1

# -------------------
# Agent definition
# -------------------
include("agents.jl")

# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SR, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)
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

    metricsArray, eval_metricsArray = SeaPearl.train!(
    valueSelectionArray=valueSelectionArray,
    generator=tsptw_generator,
    nbEpisodes=nbEpisodes,
    strategy=SeaPearl.DFSearch(),
    variableHeuristic=variableSelection,
    out_solver=true,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(valueSelectionArray,tsptw_generator; evalFreq = evalFreq, nbInstances = nbInstances)
)

    trained_weights = params(agent.policy.learner.approximator.model)
    @save dir*"/model_weights_tsptw"*string(n_city)*".bson" trained_weights

    SeaPearlExtras.storedata(metricsArray[1]; filename=dir*"/tsptw_$(n_city)_training")
    SeaPearlExtras.storedata(eval_metricsArray[:,1]; filename=dir*"/tsptw_$(n_city)_trained")
    SeaPearlExtras.storedata(eval_metricsArray[:,2]; filename=dir*"/tsptw_$(n_city)_nearest")
    for i = 1:nbRandomHeuristics
        SeaPearlExtras.storedata(eval_metricsArray[:,i+2]; filename=dir*"/tsptw_$(n_city)_random$(i)")
    end

    return metricsArray, eval_metricsArray
end


metricsArray, eval_metricsArray = trytrain(nbEpisodes)
nothing
