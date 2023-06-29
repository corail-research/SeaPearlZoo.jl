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

mis_settings, instance_generator, csv_path, save_model, device = set_settings()

if device == gpu
    CUDA.device!(numDevice)
end

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

output_size = instance_generator.n
agent_config = MisAgentConfig(0.99f0, 64, output_size, 4, 400, 1, 100, 20000)

# pool = SeaPearl.meanPooling()
trajectory_capacity = agent_config.trajectory_capacity
batch_size = agent_config.batch_size
seedEval = 123
update_freq = agent_config.update_freq
target_update_freq= agent_config.target_update_freq
n_step_per_episode = Int(round(mis_settings.nbNewVertices//2))+mis_settings.nbInitialVertices

update_horizon = Int(round(n_step_per_episode//2))

evalFreq=mis_settings.evalFreq
step_explorer = Int(floor(mis_settings.nbEpisodes*n_step_per_episode/2))

generator = instance_generator
eval_generator = generator

rngExp = MersenneTwister(seedEval)
init = Flux.glorot_uniform(MersenneTwister(seedEval))


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

feature_size = [6, 5, 2] 

rngExp = MersenneTwister(seedEval)
init = Flux.glorot_uniform(MersenneTwister(seedEval))

SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

agent = get_heterogeneous_agent(;
get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
get_explorer = () -> get_epsilon_greedy_explorer(step_explorer, 0.01; rng = rngExp ),
batch_size=batch_size,
update_horizon=update_horizon,
min_replay_history=Int(round(16*n_step_per_episode//2)),
update_freq=update_freq,
target_update_freq=target_update_freq,
get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
    feature_size=feature_size,
    conv_size=8,
    dense_size=16,
    output_size=1,
    n_layers_graph=3,
    n_layers_node=3,
    n_layers_output=2, 
    # pool=pool,
    σ=NNlib.leakyrelu,
    init = init,
    device = device
),
γ =  0.99f0
)

reward = SeaPearl.GeneralReward

learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)

selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
heuristic_max = SeaPearl.BasicHeuristic(selectMax)

randomHeuristics = []
for i in 1:mis_settings.nbRandomHeuristics
    push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
end

valueSelectionArray = [learned_heuristic, heuristic_max]
append!(valueSelectionArray, randomHeuristics)
variableSelection = SeaPearl.MinDomainVariableSelection{false}()

metricsArray, eval_metricsArray = solve_learning_mis(agent, agent_config, mis_settings, instance_generator, false, save_model)

save_metrics(eval_metricsArray, csv_path)
