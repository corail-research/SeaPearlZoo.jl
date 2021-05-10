  

using Revise
using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Statistics
using Zygote
using GeometricFlux
using Random
using BSON: @save, @load

using Plots
gr()

# -------------------
# Generator
# -------------------
knapsack_generator = SeaPearl.KnapsackGenerator(15, 10, 0.2)

# -------------------
# Internal variables
# -------------------
include("rewards.jl")
include("features.jl")
numInFeatures = SeaPearl.feature_length(knapsack_generator, SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})
state_size = SeaPearl.arraybuffer_dims(knapsack_generator, SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nb_episodes = 2000
eval_freq = 100
nb_instances = 10
nb_random_heuristics = 0

# -------------------
# Agent definition
# -------------------
include("agents.jl")


# -------------------
# Variable/Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{KnapsackFeaturization}, knapsackReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)
basicHeuristic = SeaPearl.BasicHeuristic((x; cpmodel=nothing) -> SeaPearl.maximum(x.domain)) # Basic value-selection heuristic

struct KnapsackVariableSelection <: SeaPearl.AbstractVariableSelection{false} end
function (::KnapsackVariableSelection)(model::SeaPearl.CPModel)
    i = 1
    while SeaPearl.isbound(model.variables["x[" * string(i) * "]"])
        i += 1
    end
    return model.variables["x[" * string(i) * "]"]
end

# -------------------
# Metrics definition
# -------------------
function GenerateMetricsArray(nepisodes::Int)
    global meanNodeVisited = Array{Float32}(undef, nepisodes)
    global meanNodeVisitedBasic = Array{Float32}(undef, nepisodes)
    global nodeVisitedBasic = Array{Int64}(undef, nepisodes)
    global nodeVisitedLearned = Array{Int64}(undef, nepisodes)
    global meanOver = 10 #range of the shifting mean
end

"""
    function metricsFun(;kwargs...)

This function retrieve the number of node visited for a given heuristic be it Learned or Basic, at the i-th episode
and compute a slidding mean over episodes. 
"""
function metricsFun(;kwargs...)
    i = kwargs[:episode]
    if kwargs[:heuristic] == learnedHeuristic
        currentNodeVisited = kwargs[:nodeVisited]
        nodeVisitedLearned[i] = currentNodeVisited

        currentMean = 0.
        if i <= meanOver
            currentMean = mean(nodeVisitedLearned[1:i])
        else
            currentMean = mean(nodeVisitedLearned[(i-meanOver+1):i])
        end
        meanNodeVisited[i] = currentMean
    else
        currentNodeVisited = kwargs[:nodeVisited]
        nodeVisitedBasic[i] = currentNodeVisited

        currentMean = 0.
        if i <= meanOver
            currentMean = mean(nodeVisitedBasic[1:i])
        else
            currentMean = mean(nodeVisitedBasic[(i-meanOver+1):i])
        end
        meanNodeVisitedBasic[i] = currentMean
    end
end

# -------------------
# Core function
# -------------------
function trytrain(nepisodes::Int)

    bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded = SeaPearl.train!(
        valueSelectionArray=[learnedHeuristic, basicHeuristic], 
        generator=knapsack_generator,
        nb_episodes=nepisodes,
        strategy=SeaPearl.DFSearch,
        variableHeuristic=KnapsackVariableSelection(),
        metricsFun=metricsFun,
        verbose=true, #true to print processus
        evaluator=SeaPearl.SameInstancesEvaluator(eval_freq=eval_freq, nb_instances=nb_instances)
    )
    #println(bestsolutions)
    println(convert(Array{Int},nodevisited))
    
    #saving model weights
    trained_weights = params(approximator_model)
    @save "model_weights_knapsack"*string(knapsack_generator.nb_items)*".bson" trained_weights

    # plot 
    max_y =1.1*maximum([maximum(nodeVisitedLearned),maximum(nodeVisitedBasic)])
    p = plot(1:nepisodes, 
            [nodeVisitedLearned[1:nepisodes] meanNodeVisited[1:nepisodes] nodeVisitedBasic[1:nepisodes] meanNodeVisitedBasic[1:nepisodes] (nodeVisitedLearned-nodeVisitedBasic)[1:nepisodes] (meanNodeVisited-meanNodeVisitedBasic)[1:nepisodes]], 
            xlabel="Episode", 
            ylabel="Number of nodes visited", 
            label = ["Learned" "mean/$meanOver Learned" "Basic" "mean/$meanOver Basic" "Delta" "Mean Delta"],
            ylims = (-50,max_y)
            )
    display(p)
    
    #return bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded
end


# -------------------
# Main() definition
# -------------------
function main() 
    GenerateMetricsArray(nb_episodes)
    trytrain(nb_episodes)
 end
 
 main()

