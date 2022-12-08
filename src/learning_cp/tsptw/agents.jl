function build_tsptw_agent(num_input_features::Int)
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator = RL.NeuralNetworkApproximator(
                    model = SeaPearl.VariableOutputCPNN(
                        graphChain = Flux.Chain(
                            SeaPearl.EdgeFtLayer(num_input_features => 32, 1 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32,  4 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32,  4 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32,  4 => 4),

                        ),
                        nodeChain = Flux.Chain(
                            Flux.Dense(32, 32, relu),
                            Flux.Dense(32, 32, relu),
                        ),
                        outputChain = Flux.Dense(64, 1),
                    ),
                    optimizer = ADAM(0.0001f0)
                ),
                target_approximator = RL.NeuralNetworkApproximator(
                    model = SeaPearl.VariableOutputCPNN(
                        graphChain = Flux.Chain(
                            SeaPearl.EdgeFtLayer(num_input_features => 32, 1 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32, 4 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32, 4 => 4),
                            SeaPearl.EdgeFtLayer(32 => 32, 4 => 4),
                        ),
                        nodeChain = Flux.Chain(
                            Flux.Dense(32, 32, relu),
                            Flux.Dense(32, 32, relu),
                        ),
                        outputChain = Flux.Dense(64, 1),
                    ),
                    optimizer = ADAM(0.0001f0)
                ),
                loss_func = Flux.Losses.huber_loss,
                stack_size = nothing,
                γ = 0.99f0,
                batch_size = 1,
                update_horizon = 1,
                min_replay_history = 1,
                update_freq = 1,
                target_update_freq = 100,
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.01,
                kind = :exp,
                ϵ_init = 1.0,
                warmup_steps = 0,
                decay_steps = 2000,
                step = 1,
                is_break_tie = false,
                rng = MersenneTwister(33)
            )
        ),
        trajectory = RL.CircularArraySARTTrajectory(
            capacity = 4000,
            state = SeaPearl.TsptwTrajectoryState[] => ()
        )
    )
    return agent
end