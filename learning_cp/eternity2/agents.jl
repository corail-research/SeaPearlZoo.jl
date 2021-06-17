# Model definition
n = eternity2_generator.n
m = eternity2_generator.m
# Model definition
 approximator_model = SeaPearl.CPNN(
     graphChain = Flux.Chain(
         GeometricFlux.GraphConv(numInFeatures=>10, relu),
         GeometricFlux.GraphConv(10=>10, relu),
     ),
     nodeChain = Flux.Chain(
         Flux.Dense(10, 20, Flux.leakyrelu),
     ),
     globalChain = Flux.Chain(
         Flux.Dense(numGlobalFeature, 20, Flux.leakyrelu),
         Flux.Dense(20, 20, Flux.leakyrelu),
     ),

     outputLayer = Flux.Dense(40, n*m)
 )
 target_approximator_model = SeaPearl.CPNN(
     graphChain = Flux.Chain(
         GeometricFlux.GraphConv(numInFeatures=>10, relu),
         GeometricFlux.GraphConv(10=>10, relu),
     ),
     nodeChain = Flux.Chain(
         Flux.Dense(10, 20, Flux.leakyrelu),
     ),
     globalChain = Flux.Chain(
         Flux.Dense(numGlobalFeature, 20, Flux.leakyrelu),
         Flux.Dense(20, 20, Flux.leakyrelu),
     ),

     outputLayer = Flux.Dense(40, n*m)
 )


 approximator_model2 = SeaPearl.CPNN(
     graphChain = Flux.Chain(),
     nodeChain = Flux.Chain(
         Flux.Dense(numInFeatures, 10, Flux.relu),
         Flux.Dense(10, 10, Flux.relu)
     ),
     outputLayer = Flux.Dense(10, n*m)
 )


 target_approximator_model2 = SeaPearl.CPNN(
     graphChain = Flux.Chain(),
     nodeChain = Flux.Chain(
         Flux.Dense(numInFeatures, 10, Flux.relu),
         Flux.Dense(10, 10, Flux.relu)
     ),
     outputLayer = Flux.Dense(10, n*m)
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
            batch_size = 8, #32,
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
        legal_actions_mask = Vector{Bool} => (n*m, ),
    )
)
