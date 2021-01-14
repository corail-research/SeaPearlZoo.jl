nn_args = SeaPearl.ArgsVariableOutputGCN(numInFeatures = 6, state_rep=SeaPearl.TsptwStateRepresentation)

numInFeatures = 6

agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = SeaPearl.CPDQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.FlexVariableOutputGNN(
                    graphChain = Flux.Chain(
                        SeaPearl.EdgeFtLayer(; v_dim=numInFeatures => 32, e_dim= 1 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(32, 32, relu),
                        Flux.Dense(32, 32, relu),
                    ),
                    outputLayer = Flux.Dense(64, 1),
                    state_rep=SeaPearl.TsptwStateRepresentation
                ),
                optimizer = ADAM(0.0001f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = SeaPearl.FlexVariableOutputGNN(
                    graphChain = Flux.Chain(
                        SeaPearl.EdgeFtLayer(; v_dim=numInFeatures => 32, e_dim= 1 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                        SeaPearl.EdgeFtLayer(; v_dim=32 => 32, e_dim= 4 => 4),
                    ),
                    nodeChain = Flux.Chain(
                        Flux.Dense(32, 32, relu),
                        Flux.Dense(32, 32, relu),
                    ),
                    outputLayer = Flux.Dense(64, 1),
                    state_rep=SeaPearl.TsptwStateRepresentation
                ),
                optimizer = ADAM(0.0001f0)
            ),
            loss_func = huber_loss,
            stack_size = nothing,
            γ = 0.99f0,
            batch_size = 1,#32,
            update_horizon = 1,
            min_replay_history = 1,
            update_freq = 1,
            target_update_freq = 100,
        ), 
        explorer = SeaPearl.CPEpsilonGreedyExplorer(
            ϵ_stable = 0.01,
            kind = :exp,
            ϵ_init = 1.0,
            warmup_steps = 0,
            decay_steps = 2000,
            step = 1,
            is_break_tie = false, 
            #is_training = true,
            seed = 33
        )
    ),
    trajectory = RL.CircularCompactSARTSATrajectory(
        capacity = 4000, 
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
