function create_agent(args::Any)

    approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(args.num_input_features => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
            
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 32, Flux.leakyrelu),
        ),
        outputChain = Flux.Dense(32, 2),
    )
    target_approximator_model = SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(args.num_input_features => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
            SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
            
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(32, 32, Flux.leakyrelu),
            Flux.Dense(32, 32, Flux.leakyrelu),
        ),
        outputChain = Flux.Dense(32, 2),
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
                γ = 0.99f0,
                batch_size = 16, #32,
                update_horizon = 5, #what if the number of nodes in a episode is smaller
                min_replay_history = 128,
                update_freq = 1,
                target_update_freq = 200,
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.1,
                kind = :exp,
                decay_steps = 1000,
                step = 1,
                #rng = rng
            )
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = 2000,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (2, ),
        )
    )
    return agent
end