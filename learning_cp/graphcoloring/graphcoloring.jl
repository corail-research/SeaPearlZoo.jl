using SeaPearl
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using Random

using Plots
gr()


# -------------------
# Generator
# -------------------
coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(20, 5, 0.5)


include("rewards.jl")
include("features.jl")

# -------------------
# Internal variables
# -------------------
numInFeatures = SeaPearl.feature_length(coloring_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
state_size = SeaPearl.arraybuffer_dims(coloring_generator, SeaPearl.DefaultStateRepresentation{BetterFeaturization})
maxNumberOfCPNodes = state_size[1]

# -------------------
# Experience variables
# -------------------
nb_episodes = 600
eval_freq = 30
nb_instances = 10
nb_random_heuristics = 10

# -------------------
# Agent definition
# -------------------
include("agents.jl")
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

# -------------------
# Variable selection
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()



# -------------------
# -------------------
# TRAINING
# -------------------
# -------------------
valueSelectionArray = [learnedHeuristic, heuristic_min]
append!(valueSelectionArray, randomHeuristics)

bestsolutions, nodevisited, timeneeded, eval_nodes, eval_tim = SeaPearl.train!(
    valueSelectionArray=valueSelectionArray, 
    generator=coloring_generator,
    nb_episodes=nb_episodes,
    strategy=SeaPearl.DFSearch,
    variableHeuristic=variableSelection,
    out_solver=false,
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

CSV.write("training_gc_"*string(coloring_generator.n)*".csv", df_training)

# -------------------
# Benchmarking
# -------------------
benchmark_nodes, benchmark_time = SeaPearl.benchmark_solving(;
    valueSelectionArray=valueSelectionArray, 
    generator=coloring_generator,
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
CSV.write("benchmark_gc_"*string(coloring_generator.n)*".csv", df_benchmark)

