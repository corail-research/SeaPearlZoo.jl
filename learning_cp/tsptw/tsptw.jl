using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using Statistics

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
state_size = (n_city, n_city+6+2)

# -------------------
# Experience variables
# -------------------
n_episodes = 3001
eval_freq = 250
nb_instances = 5

# -------------------
# Agent definition
# -------------------
include("agents.jl")
learnedHeuristic = SeaPearl.LearnedHeuristic{SeaPearl.TsptwStateRepresentation, SeaPearl.TsptwReward, SeaPearl.VariableOutput}(agent)

# Basic value-selection heuristic
include("nearest_heuristic.jl")
nearest_heuristic = SeaPearl.BasicHeuristic(select_nearest_neighbor)

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

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=[learnedHeuristic, nearest_heuristic], 
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
end

CSV.write("training_tsptw_"*string(n_city)*".csv", df_training)

# -------------------
# Benchmarking
# -------------------
benchmark_nodes, benchmark_time = SeaPearl.benchmark_solving(;
    valueSelectionArray=[learnedHeuristic, nearest_heuristic], 
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
CSV.write("benchmark_tsptw_"*string(n_city)*".csv", df_benchmark)


