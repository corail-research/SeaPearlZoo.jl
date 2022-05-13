trajectory_capacity = 3000

agent = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 4, Flux.leakyrelu),
                        SeaPearl.GraphConv(4 => 4, Flux.leakyrelu),
                        SeaPearl.GraphConv(4 => 4, Flux.leakyrelu)
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(4, 8, Flux.leakyrelu),
                        Flux.Dense(8, 4, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(4, 4, Flux.leakyrelu),
                        Flux.Dense(4, n_city),
                    )),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeatures => 4, Flux.leakyrelu),
                        SeaPearl.GraphConv(4 => 4, Flux.leakyrelu),
                        SeaPearl.GraphConv(4 => 4, Flux.leakyrelu)
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(4, 8, Flux.leakyrelu),
                        Flux.Dense(8, 4, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(4, 4, Flux.leakyrelu),
                        Flux.Dense(4, n_city),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=8,
            update_horizon=4,
            min_replay_history=128,
            update_freq=1,
            target_update_freq=200,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.01,
            kind = :exp,
            decay_steps=3000,
            step=1,
            #rng = rng
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=trajectory_capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_city,),
    )
)

struct HeterogeneousModel
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
    layer4::SeaPearl.HeterogeneousGraphConv
end

function HeterogeneousModel()
    layer1 = SeaPearl.HeterogeneousGraphConvInit(numInFeatures2, 4, Flux.leakyrelu)
    layer2 = SeaPearl.HeterogeneousGraphConv(4 => 4, numInFeatures2, Flux.leakyrelu)
    layer3 = SeaPearl.HeterogeneousGraphConv(4 => 4, numInFeatures2, Flux.leakyrelu)
    layer4 = SeaPearl.HeterogeneousGraphConv(4 => 4, numInFeatures2, Flux.leakyrelu)
    return HeterogeneousModel(layer1,layer2,layer3,layer4)
end

function (m::HeterogeneousModel)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    out4 = m.layer4(out3, original_fg)
    return out4
end

Flux.@functor HeterogeneousModel

agent2 = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(4, 8, Flux.leakyrelu),
                        Flux.Dense(8, 4, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(4, 4, Flux.leakyrelu),
                        Flux.Dense(4, n_city),
                    )),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(4, 8, Flux.leakyrelu),
                        Flux.Dense(8, 4, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(4, 4, Flux.leakyrelu),
                        Flux.Dense(4, n_city),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=8,
            update_horizon = 4,
            min_replay_history=128,
            update_freq=1,
            target_update_freq=200,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.01,
            kind = :exp,
            decay_steps=3000,
            step=1,
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=trajectory_capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_city,),
    )
)