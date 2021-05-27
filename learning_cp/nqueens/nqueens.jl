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

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(nqueens_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
state_size = SeaPearl.arraybuffer_dims(nqueens_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nb_episodes = 100
eval_freq = 30
nb_instances = 4
nb_random_heuristics = 0

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
for i in 1:nb_random_heuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learnedHeuristic, heuristic_min]
append!(valueSelectionArray, randomHeuristics)
# -------------------
# Variable Heuristic definition
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()

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

    bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded  = SeaPearl.train!(
        valueSelectionArray=valueSelectionArray,
        generator=coloring_generator,
        nb_episodes=nb_episodes,
        strategy=SeaPearl.DFSearch,
        variableHeuristic=variableSelection,
        metricsFun=metricsFun,
        out_solver=false,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = nb_instances)
    )

    #saving model weights
    trained_weights = params(approximator_model)
    @save "model_weights_gc"*string(coloring_generator.n)*".bson" trained_weights

    return bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded
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
    CSV.write("training_gc_"*string(coloring_generator.n)*".csv", df_training)
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
            ylims = (0,max_y)
            )
    display(p)
    savefig(p,"node_visited_graphcoloring_$(coloring_generator.n).png")
end

# -------------------
# Benchmarking
# One can compare the performance of different Heuristic, be it classic deterministic ones, or learned one using a specified agent and instance generator
# -------------------
function benchmark()
    benchmark_nodes, benchmark_time = SeaPearl.benchmark_solving(;
        valueSelectionArray=valueSelectionArray,
        generator=coloring_generator,
        strategy=SeaPearl.DFSearch,
        variableHeuristic=variableSelection,
        out_solver=false,
        verbose = false,
        evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = nb_instances)
    )

    df_benchmark = DataFrame()
    df_benchmark.nodes_learned = benchmark_nodes[1, :]
    df_benchmark.time_learned = benchmark_time[1, :]
    df_benchmark.nodes_basic = benchmark_nodes[2, :]
    df_benchmark.time_basic = benchmark_time[2, :]
    for k in 1:nb_random_heuristics
        df_benchmark[!, "nodes_random_"*string(k)] = benchmark_nodes[2+k, :]
        df_benchmark[!, "time_random_"*string(k)] = benchmark_time[2+k, :]
    end
    CSV.write("benchmark_gc_"*string(coloring_generator.n)*".csv", df_benchmark)
end


# -------------------
# -------------------

GenerateMetricsArray(nb_episodes)
bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded=trytrain(nb_episodes)
store_training_data(eval_nodevisited, eval_timeneeded)
plot_result()
benchmark()
