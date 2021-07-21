# Model definition
n = eternity2_generator.n
m = eternity2_generator.m

activation = Flux.leakyrelu
approximator_GNN = GeometricFlux.GraphConv(64 => 64, activation)
target_approximator_GNN = GeometricFlux.GraphConv(64 => 64, activation)
gnnlayers = 1

approximator_model = SeaPearl.FullFeaturedCPNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures=>12, activation),GeometricFlux.GraphConv(12=>12, activation)
    ),
    nodeChain = Flux.Chain(
    Flux.Dense(12, 12, activation),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(24, 32, activation),
        Flux.Dense(32, 1),
    ),
) #|> gpu
target_approximator_model = SeaPearl.FullFeaturedCPNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures=>12, activation),GeometricFlux.GraphConv(12=>12, activation)
    ),
    nodeChain = Flux.Chain(
    Flux.Dense(12, 12, activation),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(24, 32, activation),
        Flux.Dense(32, 1),
    ),
) #|> gpu

"""
if isfile("model_weights_gc"*string(nqueens_generator.board_size)*".bson")
    println("Parameters loaded from ", "model_weights_gc"*string(nqueens_generator.board_size)*".bson")
    @load "model_weights_gc"*string(nqueens_generator.board_size)*".bson" trained_weights
    Flux.loadparams!(approximator_model, trained_weights)
    Flux.loadparams!(target_approximator_model, trained_weights)
end
"""
#rng = MersenneTwister(33)

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
            batch_size = 8, #32,
            update_horizon = 4,
            min_replay_history = 8,
            update_freq = 8,
            target_update_freq = 100,
            #rng = rng,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.1,
            #kind = :exp,
            decay_steps = nbEpisodes - 100,
            step = 1,
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 200,
        state = SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask = Vector{Bool} => (n*m, ),
    )
)
