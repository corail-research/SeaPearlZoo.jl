using Flux
using LightGraphs
using Random
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl

include("../coloring_config.jl")
include("heterogeneous_coloring_model.jl")
include("../coloring_pipeline.jl")
include("argparse_heterogeneous_graph_coloring.jl")
include("../../utils/save_metrics.jl")


# -------------------
# Get and set the arguments
# -------------------
coloring_settings, instance_generator, eval_generator, csv_path, save_model, device = set_settings()
output_size = instance_generator.n
agent_config = ColoringAgentConfig(0.99f0, 32, output_size, 4, 400, 1, 350, 3000)

n_nodes = coloring_settings.nbNodes
seedEval = 42
decay_steps = Int(floor(coloring_settings.nbEpisodes*coloring_settings.restartPerInstances*(n_nodes+1)*0.5))
rngExp = MersenneTwister(seedEval)

chosen_features = Dict(
    "node_number_of_neighbors" => true,
    "constraint_type" => true,
    "constraint_activity" => true,
    "nb_not_bounded_variable" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "variable_is_objective" => true,
    "variable_assigned_value" => true,
    "variable_is_bound" => true,
    "values_raw" => true)

if device == gpu
    CUDA.device!(numDevice)
end

# -------------------
# Value Heuristic definition
# -------------------
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

# -------------------
# Set the reward
# -------------------
reward = SeaPearl.GeneralReward

# -------------------
# Set the State Representation
# -------------------
SR_ffcpnn = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Build the agent
# -------------------
approximator_model = build_graph_coloring_approximator_model(cpu)
target_aproximator_model = build_graph_coloring_target_approximator_model(cpu)
agent_ffcpnn = build_graph_coloring_agent(approximator_model, target_aproximator_model, agent_config, rngExp, decay_steps)

# -------------------
# Learned value selection heuristic
# -------------------
learned_heuristic_ffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_ffcpnn,reward,SeaPearl.FixedOutput}(agent_ffcpnn, chosen_features=chosen_features)

# -------------------
# Basic value-selection heuristics
# -------------------
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

randomHeuristics = []
for i in 1:coloring_settings.nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learned_heuristic_ffcpnn, heuristic_min]
append!(valueSelectionArray, randomHeuristics)

# -------------------
# Variable Heuristic definition
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}() 

# -------------------
# Solve the instances
# -------------------
metricsArray, eval_metricsArray = solve_learning_coloring(agent_ffcpnn, agent_config, coloring_settings, instance_generator, eval_generator, false, save_model)

# -------------------
# Save the metrics
# -------------------
save_metrics(eval_metricsArray, csv_path)