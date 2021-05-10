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

include("rewards.jl")
include("features.jl")

# -------------------
# Generator
# -------------------
knapsack_generator = SeaPearl.KnapsackGenerator(15, 10, 0.2)

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(knapsack_generator, SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})
state_size = SeaPearl.arraybuffer_dims(knapsack_generator, SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nb_episodes = 10
eval_freq = 100
nb_instances = 10
nb_random_heuristics = 0

# -------------------
# Agent definition
# -------------------
include("agents.jl")


# -------------------
# Value Heuristic definition
# -------------------
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.DefaultStateRepresentation{KnapsackFeaturization}, knapsackReward, SeaPearl.FixedOutput}(agent, maxNumberOfCPNodes)
basicHeuristic = SeaPearl.BasicHeuristic((x; cpmodel=nothing) -> SeaPearl.maximum(x.domain)) # Basic value-selection heuristic

# -------------------
# Variable Heuristic definition
# ------------------- 
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
function GenerateMetricsArray(nb_episodes::Int)
    global meanNodeVisited = Array{Float32}(undef, nb_episodes)
    global meanNodeVisitedBasic = Array{Float32}(undef, nb_episodes)
    global nodeVisitedBasic = Array{Int64}(undef, nb_episodes)
    global nodeVisitedLearned = Array{Int64}(undef, nb_episodes)
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
# -------------------
# Core function
# -------------------
# -------------------
function trytrain(nb_episodes::Int)

    bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded = SeaPearl.train!(
        valueSelectionArray=[learnedHeuristic, basicHeuristic], 
        generator=knapsack_generator,
        nb_episodes=nb_episodes,
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
    
    return bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded
end


# -------------------
# Plotting function
# -------------------
function plot_result()
    max_y =1.1*maximum([maximum(nodeVisitedLearned),maximum(nodeVisitedBasic)])
    p = plot(1:nb_episodes, 
            [nodeVisitedLearned[1:nb_episodes] meanNodeVisited[1:nb_episodes] nodeVisitedBasic[1:nb_episodes] meanNodeVisitedBasic[1:nb_episodes] (nodeVisitedLearned-nodeVisitedBasic)[1:nb_episodes] (meanNodeVisited-meanNodeVisitedBasic)[1:nb_episodes]], 
            xlabel="Episode", 
            ylabel="Number of nodes visited", 
            label = ["Learned" "mean/$meanOver Learned" "Basic" "mean/$meanOver Basic" "Delta" "Mean Delta"],
            ylims = (-50,max_y)
            )
    display(p)
    savefig(p,"node_visited_knapsack_$(knapsack_generator.nb_items).png")
end

# -------------------
# Storing training data
# -------------------
function store_training_data(eval_nodevisited::Array{Float64,3}, eval_timeneeded::Array{Float64,3})
    df_training = DataFrame()
    df_basic = DataFrame()
    for i in 1:nb_instances
        df_training[!, string(i)*"_nodes_trained"] = eval_nodevisited[:, 1, i]
        df_training[!, string(i)*"_nodes_basic"] = eval_nodevisited[:, 2, i]
        df_training[!, string(i)*"_time_trained"] = eval_timeneeded[:, 1, i]
        df_training[!, string(i)*"_time_basic"] = eval_timeneeded[:, 2, i]
        for k in 1:nb_random_heuristics
            df_training[!, string(i)*"_nodes_random_"*string(k)] = eval_nodevisited[:, 2+k, i]
            df_training[!, string(i)*"_time_basic_"*string(k)] = eval_timeneeded[:, 2+k, i]
        end
    end
    CSV.write("training_knapsack_"*string(knapsack_generator.nb_items)*".csv", df_training)
end


# -------------------
# -------------------

GenerateMetricsArray(nb_episodes)
bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded=trytrain(nb_episodes)
store_training_data(eval_nodevisited, eval_timeneeded)
plot_result()


