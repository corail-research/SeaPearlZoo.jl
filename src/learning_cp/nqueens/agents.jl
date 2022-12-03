include("model_config.jl")


function build_model(model_config::ModelConfig)
    model = SeaPearl.CPNN(
        graphChain=Flux.Chain(
            SeaPearl.GraphConv(model_config.num_input_features => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
        ),
        nodeChain=Flux.Chain(
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 16, Flux.leakyrelu),
        ),
        globalChain=Flux.Chain(),
        outputChain=Flux.Chain(
            Flux.Dense(16, 32, Flux.leakyrelu),
            Flux.Dense(32, model_config.board_size),
        )
    )
    if model_config.gpu
        return model |> gpu
    end
    return model
end

function build_agent(agent_config::AgentConfig)
    agent = RL.Agent(
        policy=RL.QBasedPolicy(
            learner=RL.DQNLearner(
                approximator=RL.NeuralNetworkApproximator(
                    model=agent_config.approximator_model,
                    optimizer=ADAM()
                ),
                target_approximator=RL.NeuralNetworkApproximator(
                    model=agent_config.target_approximator_model,
                    optimizer=ADAM()
                ),
                loss_func=Flux.Losses.huber_loss,
                batch_size = agent_config.batch_size,
                update_horizon = agent_config.update_horizon,
                min_replay_history = agent_config.min_replay_history,
                update_freq = agent_config.update_freq,
                target_update_freq = agent_config.target_update_freq 
            ),
            explorer=RL.EpsilonGreedyExplorer(
                ϵ_init = agent_config.ϵ_init,
                ϵ_stable = agent_config.ϵ_stable,
                kind = agent_config.kind,
                decay_steps = agent_config.decay_steps,
                step = agent_config.step 
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = agent_config.trajectory_capacity,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (agent_config.board_size, ) ,
        )
    )
    return agent
end
