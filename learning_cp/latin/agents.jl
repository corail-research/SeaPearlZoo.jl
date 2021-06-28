# Model definition
N = latin_generator.N
p = latin_generator.p
approximator_GNN = GeometricFlux.GraphConv(64 => 64, Flux.leakyrelu)
target_approximator_GNN = GeometricFlux.GraphConv(64 => 64, Flux.leakyrelu)
gnnlayers = 4

approximator_model = SeaPearl.CPNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures => 64, Flux.leakyrelu),
        [approximator_GNN for i = 1:gnnlayers]...
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(64, 64, Flux.leakyrelu),
        Flux.Dense(64, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    globalChain = Flux.Chain(
        Flux.Dense(numGlobalFeature, 64, Flux.leakyrelu),
        Flux.Dense(64, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(32, 32, Flux.leakyrelu),
        Flux.Dense(32, N),
    )) #|> gpu
target_approximator_model = SeaPearl.CPNN(
    graphChain = Flux.Chain(
        GeometricFlux.GraphConv(numInFeatures => 64, Flux.leakyrelu),
        [approximator_GNN for i = 1:gnnlayers]...
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(64, 64, Flux.leakyrelu),
        Flux.Dense(64, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    globalChain = Flux.Chain(
        Flux.Dense(numGlobalFeature, 64, Flux.leakyrelu),
        Flux.Dense(64, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(32, 32, Flux.leakyrelu),
        Flux.Dense(32, N),
    )
) #|> gpu


approximator_model2 = SeaPearl.CPNN(
    graphChain = Flux.Chain(),
    nodeChain = Flux.Chain(
        Flux.Dense(numInFeatures, 10, Flux.relu),
        Flux.Dense(10, 10, Flux.relu)
    ),
    outputChain = Flux.Dense(10, N)
)


target_approximator_model2 = SeaPearl.CPNN(
    graphChain = Flux.Chain(),
    nodeChain = Flux.Chain(
        Flux.Dense(numInFeatures, 10, Flux.relu),
        Flux.Dense(10, 10, Flux.relu)
    ),
    outputChain = Flux.Dense(10, N)
)

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
            batch_size = 16, #32,
            update_horizon = 3, #what if the number of nodes in a episode is smaller
            min_replay_history = 10,
            update_freq = 8,
            target_update_freq = 100,
            #rng = rng,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.1,
            #kind = :exp,
            decay_steps = 2000,
            step = 1,
            #rng = rng
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 1000,
        state = SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask = Vector{Bool} => (N, ),
    )
)
