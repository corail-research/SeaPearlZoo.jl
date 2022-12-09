using SeaPearl


function build_tsptw_agent(num_input_features::Int, num_cities::Int)
    # trajectory_capacity = 2000
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator = RL.NeuralNetworkApproximator(
                    model = SeaPearl.CPNN(
                        graphChain = Flux.Chain(
                            SeaPearl.GraphConv(num_input_features => 32, Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        ),
                        nodeChain = Flux.Chain(
                            Flux.Dense(32, 32, relu),
                            Flux.Dense(32, 32, relu),
                        ),
                        outputChain = Flux.Chain(Flux.Dense(32, num_cities)),
                    ),
                    optimizer = ADAM(0.0005f0)
                ),
                target_approximator = RL.NeuralNetworkApproximator(
                    model = SeaPearl.CPNN(
                        graphChain = Flux.Chain(
                            SeaPearl.GraphConv(numInFeatures => 32,  Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                            SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        ),
                        nodeChain = Flux.Chain(
                            Flux.Dense(32, 32, relu),
                            Flux.Dense(32, 32, relu),
                        ),
                        outputChain = Flux.Chain(Flux.Dense(32, num_cities)),
                    ),
                    optimizer = ADAM(0.0005f0)
                ),
                loss_func = Flux.Losses.huber_loss,
                stack_size = nothing,
                γ = 0.99f0,
                batch_size = 16,
                update_horizon = 4,
                min_replay_history = 64,
                update_freq = 1,
                target_update_freq = 200,
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.1,
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
            capacity = 2000,
            state = SeaPearl.DefaultTrajectoryState[] => ()
        )
    )
    return agent
end