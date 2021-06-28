# Model definition
n = eternity2_generator.n
m = eternity2_generator.m

activation = Flux.leakyrelu

approximator_model = SeaPearl.FullFeaturedCPNN(
    graphChain = Flux.Chain(),
    nodeChain = Flux.Chain(
        Flux.Dense(numInFeatures, 64, activation),
        Flux.Dense(64, 64, activation),
        Flux.Dense(64, 64, activation),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(128, 64, activation),
        Flux.Dense(64, 1),
    )
) |> cpu
target_approximator_model = SeaPearl.FullFeaturedCPNN(
    graphChain = Flux.Chain(),
    nodeChain = Flux.Chain(
        Flux.Dense(numInFeatures, 64, activation),
        Flux.Dense(64, 64, activation),
        Flux.Dense(64, 64, activation),
    ),
    outputChain = Flux.Chain(
        Flux.Dense(128, 64, activation),
        Flux.Dense(64, 1),
    )
) |> cpu

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
            batch_size = 32, #32,
            update_horizon = 4, #what if the number of nodes in a episode is smaller
            min_replay_history = 8,
            update_freq = 8,
            target_update_freq = 100,
            #rng = rng,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.1,
            #kind = :exp,
            decay_steps = 3000,
            step = 1,
            #rng = rng
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 200,
        state = SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask = Vector{Bool} => (n*m, ),
    )
)
