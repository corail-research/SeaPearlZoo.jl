agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = SeaPearl.CPDQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.FlexGNN(
                    graphChain = Flux.Chain(
                        GeometricFlux.GCNConv(numInFeatures => 20),
                        GeometricFlux.GCNConv(20 => 20),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(20, 20),
                    ),
                    outputLayer = Flux.Dense(20, 10)
                ),
                optimizer = ADAM(0.0005f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.FlexGNN(
                    graphChain = Flux.Chain(
                        GeometricFlux.GCNConv(numInFeatures => 20),
                        GeometricFlux.GCNConv(20 => 20),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(20, 20),
                    ),
                    outputLayer = Flux.Dense(20, 10)
                ),
                optimizer = ADAM(0.0005f0)
            ),
            loss_func = huber_loss,
            stack_size = nothing,
            γ = 0.9999f0,
            batch_size = 1, #32,
            update_horizon = 25,
            min_replay_history = 1,
            update_freq = 10,
            target_update_freq = 200,
            seed = 22,
        ), 
        explorer = SeaPearl.CPEpsilonGreedyExplorer(
            ϵ_stable = 0.001,
            kind = :exp,
            ϵ_init = 1.0,
            warmup_steps = 0,
            decay_steps = 1000,
            step = 1,
            is_break_tie = false, 
            #is_training = true,
            seed = 33
        )
    ),
    trajectory = RL.CircularCompactSARTSATrajectory(
        capacity = 3000, 
        state_type = Float32, 
        state_size = state_size,#(46, 93, 1),
        action_type = Int,
        action_size = (),
        reward_type = Float32,
        reward_size = (),
        terminal_type = Bool,
        terminal_size = ()
    ),
    role = :DEFAULT_PLAYER
)
