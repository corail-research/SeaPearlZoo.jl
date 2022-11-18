include("model_config.jl")


function build_knapsack_approximator_model(approximator_config::KnapsackApproximatorConfig)
    approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(approximator_config.num_input_features => 16, Flux.leakyrelu),
            [approximator_config.GNN_layer for i = 1: approximator_config.num_GNN_layers]...
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Dense(16, 2),
    )
    if approximator_config.gpu
        return approximator_model |> gpu    
    else
        return approximator_model
    end
end

function build_knapsack_target_approximator_model(approximator_config::KnapsackApproximatorConfig)
    target_approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(approximator_config.num_input_features => 16, Flux.leakyrelu),
            [approximator_config.GNN_layer for i = 1: approximator_config.num_GNN_layers]...
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Dense(16, 2),
    )
    if approximator_config.gpu
        return target_approximator_model |> gpu    
    else
        return target_approximator_model
    end
end

function build_knapsack_agent(
    approximator_model::SeaPearl.CPNN, 
    target_approximator_model::SeaPearl.CPNN,
    knapsack_agent_config::KnapsackAgentConfig
    )
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator = RL.NeuralNetworkApproximator(
                    model = approximator_model,
                    optimizer = ADAM()
                ),
                target_approximator = RL.NeuralNetworkApproximator(
                    model = target_approximator_model,
                    optimizer = ADAM()
                ),
                loss_func = Flux.Losses.huber_loss,
                γ = knapsack_agent_config.gamma,
                batch_size = knapsack_agent_config.batch_size,
                update_horizon = knapsack_agent_config.update_horizon,
                min_replay_history = knapsack_agent_config.min_replay_history,
                update_freq = knapsack_agent_config.update_freq,
                target_update_freq = knapsack_agent_config.target_update_freq
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.01,
                decay_steps = knapsack_agent_config.num_episodes,
                step = 1,
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = 1000,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (2, ),
        )
    )
    return agent
end