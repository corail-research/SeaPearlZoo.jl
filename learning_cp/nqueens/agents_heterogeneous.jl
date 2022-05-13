trajectory_capacity = 3000

agent = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 8, Flux.leakyrelu),
                        SeaPearl.GraphConv(8 => 8, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, 8, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, nqueens_generator.board_size),
                    )),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 8, Flux.leakyrelu),
                        SeaPearl.GraphConv(8 => 8, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, 8, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, nqueens_generator.board_size),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=16,
            update_horizon=10,
            min_replay_history=164,
            update_freq=1,
            target_update_freq=200,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.05,
            kind = :exp,
            decay_steps=3000,
            step=1,
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=trajectory_capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (nqueens_generator.board_size,),
    )
)

struct HeterogeneousModel
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
end

function HeterogeneousModel()
    layer1 = SeaPearl.HeterogeneousGraphConvInit(numInFeatures2, 8, Flux.leakyrelu)
    layer2 = SeaPearl.HeterogeneousGraphConv(8 => 8, numInFeatures2, Flux.leakyrelu)
    layer3 = SeaPearl.HeterogeneousGraphConv(8 => 8, numInFeatures2, Flux.leakyrelu)
    return HeterogeneousModel(layer1,layer2,layer3)
end

function (m::HeterogeneousModel)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    return out3
end

Flux.@functor HeterogeneousModel

agent2 = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(24, 8, Flux.leakyrelu),
                        Flux.Dense(8, 8, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, nqueens_generator.board_size),
                    )),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(24, 8, Flux.leakyrelu),
                        Flux.Dense(8, 8, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(8, 8, Flux.leakyrelu),
                        Flux.Dense(8, nqueens_generator.board_size),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=16,
            update_horizon = 10,
            min_replay_history=164,
            update_freq=1,
            target_update_freq=200,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.05,
            kind = :exp,
            decay_steps=3000,
            step=1,
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=trajectory_capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (nqueens_generator.board_size,),
    )
)