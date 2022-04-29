# Model definition
n = coloring_generator.n

approximator_model = SeaPearl.CPNN(
    graphChain = Flux.Chain(
        SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
        SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
        SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(16, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    #globalChain = Flux.Chain(
    ##    Flux.Dense(numGlobalFeature, 64, Flux.leakyrelu),
    #    Flux.Dense(64, 32, Flux.leakyrelu),
    #    Flux.Dense(32, 16, Flux.leakyrelu),
    #),
    outputChain = Flux.Chain(
        Flux.Dense(16, 16, Flux.leakyrelu),
        Flux.Dense(16, n),
    )) #|> gpu
target_approximator_model = SeaPearl.CPNN(
    graphChain = Flux.Chain(
        SeaPearl.GraphConv(numInFeatures => 16, Flux.leakyrelu),
        SeaPearl.GraphConv(16 => 16, Flux.leakyrelu),
        SeaPearl.GraphConv(16 => 16, Flux.leakyrelu)
    ),
    nodeChain = Flux.Chain(
        Flux.Dense(16, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    #globalChain = Flux.Chain(
    ##    Flux.Dense(numGlobalFeature, 64, Flux.leakyrelu),
    #    Flux.Dense(64, 32, Flux.leakyrelu),
    #    Flux.Dense(32, 16, Flux.leakyrelu),
    #),
    outputChain = Flux.Chain(
        Flux.Dense(16, 16, Flux.leakyrelu),
        Flux.Dense(16, n),
    ) #|> gpu
) #|> gpu


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
            γ = 0.99f0,
            batch_size = 8, #32,
            update_horizon = 10, #what if the number of nodes in a episode is smaller
            min_replay_history = 8,
            update_freq = 1,
            target_update_freq = 200,
        ),
        explorer = RL.EpsilonGreedyExplorer(
            ϵ_stable = 0.1,
            kind = :exp,
            decay_steps = 2000,
            step = 1,
            #rng = rng
        )
    ),
    trajectory = RL.CircularArraySLARTTrajectory(
        capacity = 1000,
        state = SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask = Vector{Bool} => (n, ),
    )
)
