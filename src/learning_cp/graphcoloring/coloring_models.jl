# Model definition
include("coloring_config.jl")

function build_graph_coloring_approximator_model(output_size:: Int)
    approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(16, 16, Flux.leakyrelu),
            Flux.Dense(16, output_size),
        )) #|> gpu
    return approximator_model
end
function build_graph_coloring_target_approximator_model(output_size:: Int)
    target_approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
            SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(16, 16, Flux.leakyrelu),
            Flux.Dense(16, output_size),
        ) #|> gpu
    ) #|> gpu
    return target_approximator_model
end

function build_graph_coloring_agent(approximator_model, target_approximator_model, agent_config :: ColoringAgentConfig)
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
                γ = agent_config.gamma,
                batch_size = agent_config.batch_size,
                update_horizon = agent_config.update_horizon,
                min_replay_history = agent_config.min_replay_history,
                update_freq = agent_config.update_freq,
                target_update_freq = agent_config.target_update_freq
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.01,
                kind = :exp,
                decay_steps = 3000,
                step = 1
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = agent_config.trajectory_capacity,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (agent_config.output_size, ),
        )
    )
    return agent
end
