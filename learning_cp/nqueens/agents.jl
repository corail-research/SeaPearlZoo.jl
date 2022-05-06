trajectory_capacity = 50000

approximator_model = SeaPearl.CPNN(
    graphChain=Flux.Chain(
        SeaPearl.GraphConv(numInFeatures => 32, Flux.leakyrelu),
        SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
        SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
    ),
    nodeChain=Flux.Chain(
        Flux.Dense(32, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    globalChain=Flux.Chain(),
    outputChain=Flux.Chain(
        Flux.Dense(16, 32, Flux.leakyrelu),
        Flux.Dense(32, nqueens_generator.board_size),
    )
)
#|> gpu
target_approximator_model = SeaPearl.CPNN(
    graphChain=Flux.Chain(
        SeaPearl.GraphConv(numInFeatures => 32, Flux.leakyrelu),
        SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
        SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
    ),
    nodeChain=Flux.Chain(
        Flux.Dense(32, 32, Flux.leakyrelu),
        Flux.Dense(32, 16, Flux.leakyrelu),
    ),
    globalChain=Flux.Chain(),
    outputChain=Flux.Chain(
        Flux.Dense(16, 32, Flux.leakyrelu),
        Flux.Dense(32, nqueens_generator.board_size),
    )
) #|> gpu

agent = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=approximator_model,
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=target_approximator_model,
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size = 32, #32,
            update_horizon = 12, #what if the number of nodes in a episode is smaller
            min_replay_history=256,
            update_freq=1,
            target_update_freq=200,
            #rng = rng,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_init=1.0,
            ϵ_stable = 0.1,
            kind = :exp,
            decay_steps = 50000,
            step=1,
            #rng = rng
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=trajectory_capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (nqueens_generator.board_size,),
    )
)
