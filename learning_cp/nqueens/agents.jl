# Model definition
approximator_GNN = GeometricFlux.GraphConv(32 => 32, Flux.σ)
target_approximator_GNN = GeometricFlux.GraphConv(32 => 32)
gnnlayers = 5

approximator_model = SeaPearl.FlexGNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures => 32, Flux.leakyrelu),
        SeaPearl.GraphNorm(32, Flux.leakyrelu),
        #vcat([[approximator_GNN, SeaPearl.GraphNorm(32, Flux.leakyrelu)] for i = 1:gnnlayers]...)...
    ),    nodeChain = Flux.Chain(
        Flux.Dense(32, 5, Flux.relu),
    ),
    outputLayer = Flux.Dense(5, nqueens_generator.board_size, Flux.relu)
) |> gpu
target_approximator_model = SeaPearl.FlexGNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures => 32, Flux.leakyrelu),
        SeaPearl.GraphNorm(32, Flux.leakyrelu),
        #vcat([[approximator_GNN, SeaPearl.GraphNorm(32, Flux.leakyrelu)] for i = 1:gnnlayers]...)...
    ),    nodeChain = Flux.Chain(
        Flux.Dense(32, 5, Flux.relu),
    ),
    outputLayer = Flux.Dense(5, nqueens_generator.board_size, Flux.relu)
) |> gpu

if isfile("model_weights_gc"*string(nqueens_generator.board_size)*".bson")
    println("Parameters loaded from ", "model_weights_gc"*string(nqueens_generator.board_size)*".bson")
    @load "model_weights_gc"*string(nqueens_generator.board_size)*".bson" trained_weights
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
            γ = 0.99f0,
            batch_size = 16,
            update_horizon = 6, #what if the number of nodes in a episode is smaller
            min_replay_history = 16,
            update_freq = 8,
            target_update_freq = 128,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.001,
            kind = :exp,
            ϵ_init = 1.0,
            warmup_steps = 50,
            decay_steps = 5000,
            step = 1,
            is_break_tie = false,
            #is_training = true,
            rng = MersenneTwister(33)
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 128,
        state = SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask = Vector{Bool} => (nqueens_generator.board_size, ),
    )
)
