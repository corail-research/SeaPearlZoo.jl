using SeaPearl
include("model_config.jl")


function build_approximator_model(eternity_model_config::EternityModelConfig)
    approximator_model = SeaPearl.FullFeaturedCPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(
                eternity_model_config.num_input_features=>12, 
                eternity_model_config.activation_function),
                SeaPearl.GraphConv(
                12=>12, 
                eternity_model_config.activation_function
                )
        ),
        nodeChain = Flux.Chain(
        Flux.Dense(12, 12, eternity_model_config.activation_function),
        ),
        outputChain = Flux.Chain(
            Flux.Dense(24, 32, eternity_model_config.activation_function),
            Flux.Dense(32, 1),
        ),
    )
    if eternity_model_config.gpu
        return approximator_model |> gpu
    end
    return approximator_model
end

function build_eternity2_agent(dqn_learner_config::DQNLearnerConfig, eternity_agent_config::EternityAgentConfig, num_episodes::Int)
    learner = RL.DQNLearner(
        approximator = RL.NeuralNetworkApproximator(
            model = eternity_agent_config.approximator_model,
            optimizer = ADAM()
        ),
        target_approximator = RL.NeuralNetworkApproximator(
            model = eternity_agent_config.target_approximator_model,
            optimizer = ADAM()
        ),
        loss_func = Flux.Losses.huber_loss,
        γ=dqn_learner_config.gamma,
        batch_size=dqn_learner_config.batch_size,
        update_horizon=dqn_learner_config.update_horizon,
        min_replay_history=dqn_learner_config.min_replay_history,
        update_freq=dqn_learner_config.update_freq,
        target_update_freq=dqn_learner_config.target_update_freq
    )
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner=learner,
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.01,
                decay_steps = num_episodes,
                step = 1,
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = 200,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (eternity_agent_config.n * eternity_agent_config.m, ),
        )
    )
    return agent
end