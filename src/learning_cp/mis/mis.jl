using Flux
using LightGraphs
using Random
using ReinforcementLearning
const RL = ReinforcementLearning
using SeaPearl

include("mis_config.jl")
include("mis_model.jl")
include("mis_pipeline.jl")
include("argparse_mis.jl")
include("../utils/save_metrics.jl")


# -------------------
# Get and set the arguments
# -------------------
mis_settings, instance_generator, csv_path, save_model, device, save_performance = set_settings()
output_size = instance_generator.n
agent_config = MisAgentConfig(0.99f0, 64, output_size, 4, 400, 1, 100, 20000)

seedEval = 42
n_step_per_episode = Int(round(mis_settings.nbNewVertices//2))+mis_settings.nbInitialVertices
decay_steps = Int(floor(mis_settings.nbEpisodes*n_step_per_episode/2))
eval_generator = instance_generator
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
# Set the reward
# -------------------
reward = SeaPearl.GeneralReward

# -------------------
# Set the State Representation
# -------------------
SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Build the agent
# -------------------
approximator_model = build_mis_approximator_model(cpu)
target_aproximator_model = build_mis_target_approximator_model(cpu)
agent_ffcpnn = build_mis_agent(approximator_model, target_aproximator_model, agent_config, rngExp, decay_steps)

# -------------------
# Learned value selection heuristic
# -------------------
learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnn; chosen_features=chosen_features)

# -------------------
# Basic value-selection heuristics
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

selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
heuristic_max = SeaPearl.BasicHeuristic(selectMax)

randomHeuristics = []
for i in 1:mis_settings.nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learned_heuristic, heuristic_max]
append!(valueSelectionArray, randomHeuristics)

# -------------------
# Variable Heuristic definition
# -------------------
variableSelection = SeaPearl.MinDomainVariableSelection{false}()

# -------------------
# Solve the instances
# -------------------
metricsArray, eval_metricsArray = solve_learning_mis(agent_ffcpnn, agent_config, mis_settings, instance_generator, eval_generator, false, save_model)

# -------------------
# Save the metrics
# -------------------
if save_performance
    save_metrics(eval_metricsArray, csv_path)
end