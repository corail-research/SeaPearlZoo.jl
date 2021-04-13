using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using Statistics
using Random

using Plots
gr()

# -------------------
# Generator
# -------------------
n_city = 21
grid_size = 100
max_tw_gap = 10
max_tw = 100
tsptw_generator = SeaPearl.TsptwGenerator(n_city, grid_size, max_tw_gap, max_tw, true)

# -------------------
# Internal variables
# -------------------
state_size = SeaPearl.arraybuffer_dims(tsptw_generator, SeaPearl.TsptwStateRepresentation{SeaPearl.TsptwFeaturization})
# state_size = (n_city, n_city+6+2)

# -------------------
# Experience variables
# -------------------
n_episodes = 10#3001
eval_freq = 5#250
nb_instances = 5
nb_random_heuristics = 10

# -------------------
# Agent definition
# -------------------
include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.TsptwStateRepresentation, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)

# Basic value-selection heuristic
include("nearest_heuristic.jl")
nearest_heuristic = SeaPearl.BasicHeuristic(select_nearest_neighbor)

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

# -------------------
# Variable selection
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


############# TRAIN

valueSelectionArray = [learnedHeuristic, nearest_heuristic]
append!(valueSelectionArray, randomHeuristics)

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray = valueSelectionArray, 
    generator=tsptw_generator,
    nb_episodes=n_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=true,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = eval_freq, nb_instances = nb_instances)
)

# -------------------
# Storing training data
# -------------------
using DataFrames
using CSV
df_training = DataFrame()
df_basic = DataFrame()
for i in 1:nb_instances
    df_training[!, string(i)*"_nodes_trained"] = eval_nodes[:, 1, i]
    df_training[!, string(i)*"_nodes_basic"] = eval_nodes[:, 2, i]
    df_training[!, string(i)*"_time_trained"] = eval_tim[:, 1, i]
    df_training[!, string(i)*"_time_basic"] = eval_tim[:, 2, i]
    for k in 1:nb_random_heuristics
        df_training[!, string(i)*"_nodes_random_"*string(k)] = eval_nodes[:, 2+k, i]
        df_training[!, string(i)*"_time_basic_"*string(k)] = eval_tim[:, 2+k, i]
    end
end

CSV.write("training_tsptw_"*string(n_city)*".csv", df_training)

# -------------------
# Benchmarking
# -------------------
benchmark_nodes, benchmark_time = SeaPearl.benchmark_solving(;
    valueSelectionArray=valueSelectionArray, 
    generator=tsptw_generator,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=false,
    verbose = false,
    evaluator=SeaPearl.SameInstancesEvaluator(; eval_freq = 0, nb_instances = 50)
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


