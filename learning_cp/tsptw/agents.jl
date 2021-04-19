numInFeatures = 6

# Model definition
approximator_model = SeaPearl.FlexVariableOutputGNN(
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
)
target_approximator_model = SeaPearl.FlexVariableOutputGNN(
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
)

filename = "model_weights_tsptw"*string(n_city)*".bson"
if isfile(filename)
    println("Parameters loaded from ", filename)
    @load filename trained_weights
    Flux.loadparams!(approximator_model, trained_weights)
    Flux.loadparams!(target_approximator_model, trained_weights)
end


agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = RL.DQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = approximator_model,
                optimizer = ADAM(0.0001f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = target_approximator_model,
                optimizer = ADAM(0.0001f0)
            ),
            loss_func = Flux.Losses.huber_loss,
            stack_size = nothing,
            γ = 0.99f0,
            batch_size = 1,#32,
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
            #is_training = true,
            rng = MersenneTwister(33)
        )
    ),
    trajectory = RL.CircularArraySARTTrajectory(
        capacity = 4000, 
        state = Matrix{Float32} => state_size
    )
)
