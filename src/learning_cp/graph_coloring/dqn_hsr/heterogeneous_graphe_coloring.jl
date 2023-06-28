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

coloring_settings, instance_generator, eval_generator, csv_path = set_settings()

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


reward = SeaPearl.GeneralReward
SR_ffcpnn = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

# -------------------
# Value Heuristic definition
# -------------------
output_size = instance_generator.n
agent_config = ColoringAgentConfig(0.99f0, 32, output_size, 4, 400, 1, 350, 3000)

n_nodes = coloring_settings.nbNodes
seedEval = 42
n_step_per_episode = n_nodes
update_horizon = Int(round(n_step_per_episode//2))

evalFreq=coloring_settings.evalFreq

feature_size = [6, 5, 2]

decay_steps = Int(floor(coloring_settings.nbEpisodes*coloring_settings.restartPerInstances*(n_nodes+1)*0.5))

rngExp = MersenneTwister(seedEval)
init = Flux.glorot_uniform(MersenneTwister(seedEval))
pool = SeaPearl.sumPooling()
trajectory_capacity=10000

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

agent_ffcpnn = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n_nodes),        
    get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.05; rng = rngExp ),
    batch_size=32,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=1,
    target_update_freq=7*n_step_per_episode,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=feature_size,
        conv_size=8,
        dense_size=16,
        output_size=1,
        n_layers_graph=3,
        n_layers_node=2,
        n_layers_output=2, 
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init
    ),
    γ = 0.99f0
    )

learned_heuristic_ffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_ffcpnn,reward,SeaPearl.FixedOutput}(agent_ffcpnn, chosen_features=chosen_features)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

randomHeuristics = []
for i in 1:coloring_settings.nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learned_heuristic_ffcpnn, heuristic_min]
append!(valueSelectionArray, randomHeuristics)
variableSelection = SeaPearl.MinDomainVariableSelection{false}() # Variable Heuristic definition

metricsArray, eval_metricsArray = solve_learning_coloring(agent_ffcpnn, agent_config, coloring_settings, instance_generator, eval_generator)

save_metrics(eval_metricsArray, csv_path)