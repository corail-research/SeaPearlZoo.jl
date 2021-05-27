# Model definition
approximator_model = SeaPearl.FlexGNN(
    graphChain = Flux.Chain(
        GeometricFlux.GATConv(numInFeatures => 10, heads=2, concat=true),
        GeometricFlux.GATConv(20 => 20, heads=2, concat=false),
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(20, 20),
    ),
    outputLayer = Flux.Dense(20, nqueens_generator.board_size)
)
target_approximator_model = SeaPearl.FlexGNN(
    graphChain = Flux.Chain(
        GeometricFlux.GATConv(numInFeatures => 10, heads=2, concat=true),
        GeometricFlux.GATConv(20 => 20, heads=2, concat=false),
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(20, 20),
    ),
    outputLayer = Flux.Dense(20, nqueens_generator.board_size)
)

if isfile("model_weights_gc"*string(nqueens_generatorn.board_size)*".bson")
    println("Parameters loaded from ", "model_weights_gc"*string(nqueens_generatorn.board_size)*".bson")
    @load "model_weights_gc"*string(nqueens_generatorn.board_size)*".bson" trained_weights
    Flux.loadparams!(approximator_model, trained_weights)
    Flux.loadparams!(target_approximator_model, trained_weights)
end

# Agent definition
agent = RL.Agent(
    policy = RL.QBasedPolicy(
        learner = RL.DQNLearner(
            approximator = RL.NeuralNetworkApproximator(
                model = approximator_model,
                optimizer = ADAM(0.0005f0)
            ),
            target_approximator = RL.NeuralNetworkApproximator(
                model = target_approximator_model,
                optimizer = ADAM(0.0005f0)
            ),
            loss_func = Flux.Losses.huber_loss,
            stack_size = nothing,
            γ = 0.9999f0,
            batch_size = 1, #32,
            update_horizon = 25,
            min_replay_history = 1,
            update_freq = 10,
            target_update_freq = 200,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.001,
            kind = :exp,
            ϵ_init = 1.0,
            warmup_steps = 0,
            decay_steps = 5000,
            step = 1,
            is_break_tie = false,
            #is_training = true,
            rng = MersenneTwister(33)
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 8000,
        state = Matrix{Float32} => state_size,
        legal_actions_mask = Vector{Bool} => (nqueens_generator.board_size, ),
    )
)