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
nb_episodes = 10
eval_freq = 250
nb_instances = 5
nb_random_heuristics = 5

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
for i in 1:nb_random_heuristics
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

############# TRAIN

function trytrain(nb_episodes::Int)

    bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded  = SeaPearl.train!(
    valueSelectionArray=valueSelectionArray, 
    generator=tsptw_generator,
    nb_episodes=nb_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    metricsFun=metricsFun,
    out_solver=false,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = nb_instances)
)

    #println(bestsolutions)
    println(convert(Array{Int},nodevisited))

    trained_weights = params(approximator_model)
    @save "model_weights_tsptw"*string(n_city)*".bson" trained_weights
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
    savefig(p,"node_visited_tsptw_$(tsptw_generator.n_city).png")
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
    CSV.write("training_tsptw_"*string(n_city)*".csv", df_training)
end

# -------------------
# Benchmarking
# -------------------
function benchmark()
    benchmark_nodes, benchmark_time = SeaPearl.benchmark_solving(;
    valueSelectionArray=valueSelectionArray, 
    generator=tsptw_generator,
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
    CSV.write("benchmark_tsptw_"*string(n_city)*".csv", df_benchmark)
end

# -------------------
# -------------------

GenerateMetricsArray(nb_episodes)
bestsolutions, nodevisited,timeneeded, eval_nodevisited, eval_timeneeded=trytrain(nb_episodes)
store_training_data(eval_nodevisited, eval_timeneeded)
plot_result()
benchmark()


