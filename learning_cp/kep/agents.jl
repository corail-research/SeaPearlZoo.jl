function create_agent(args::Any)
    n = args.nbNodes

    approximator_GNN = SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
    target_approximator_GNN = SeaPearl.GraphConv(32 => 32, Flux.leakyrelu)
    gnnlayers = 2

    approximator_model = SeaPearl.FullFeaturedCPNN(
    graphChain = Flux.Chain(
        SeaPearl.GraphConv(args.numInFeatures => 32, Flux.leakyrelu),
        [approximator_GNN for i = 1:gnnlayers]...
    ),
    nodeChain=Flux.Chain(
        Flux.Dense(32, 32, relu),
        Flux.Dense(32, 32, relu),
    ),
    outputChain=Flux.Dense(64, 1),
    )

    target_approximator_model = SeaPearl.FullFeaturedCPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(args.numInFeatures => 32, Flux.leakyrelu),
            [approximator_GNN for i = 1:gnnlayers]...
        ),
        nodeChain=Flux.Chain(
            Flux.Dense(32, 32, relu),
            Flux.Dense(32, 32, relu),
        ),
        outputChain=Flux.Dense(64, 1),
    )

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
                batch_size=16, #32,
                update_horizon=3, #what if the number of nodes in a episode is smaller
                min_replay_history=10,
                update_freq=8,
                target_update_freq=100,
                #rng = rng,
            ),
            explorer=RL.EpsilonGreedyExplorer(
                ϵ_stable=0.1,
                #kind = :exp,
                decay_steps=2000,
                step=1,
                #rng = rng
            )
        ),
        trajectory=RL.CircularArraySLARTTrajectory(
            capacity=1000,
            state=SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask=Vector{Bool} => (2,),
        )
    )

    return agent
end