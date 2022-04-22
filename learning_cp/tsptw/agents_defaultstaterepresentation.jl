agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = RL.DQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.CPNN(
                    graphChain = Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 32, Flux.leakyrelu),
                        SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        #SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        #SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(32, 32, relu),
                        #Flux.Dense(32, 32, relu),
                    ),
                    outputChain = Flux.Dense(32, n_city),
                ),
                optimizer = ADAM(0.0005f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.CPNN(
                    graphChain = Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 32,  Flux.leakyrelu),
                        SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        #SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                        #SeaPearl.GraphConv(32 => 32,  Flux.leakyrelu),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(32, 32, relu),
                        #Flux.Dense(32, 32, relu),
                    ),
                    outputChain = Flux.Dense(32, n_city),
                ),
                optimizer = ADAM(0.0005f0)
            ),
            loss_func = Flux.Losses.huber_loss,
            stack_size = nothing,
            γ = 0.99f0,
            batch_size = 4,#32,
            update_horizon = 1,
            min_replay_history = 32,
            update_freq = 1,
            target_update_freq = 100,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.1,
            kind = :exp,
            ϵ_init = 1.0,
            warmup_steps = 0,
            decay_steps = 2000,
            step = 1,
            is_break_tie = false,
            #is_training = true,
            rng = MersenneTwister(33)
        )
    ),
    trajectory = RL.CircularArraySARTTrajectory(
        capacity = 1000,
        state = SeaPearl.DefaultTrajectoryState[] => ()
    )
)
