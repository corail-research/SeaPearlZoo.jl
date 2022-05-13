BATCH_SIZE = @isdefined(BATCH_SIZE) ? BATCH_SIZE : 16
UPDATE_HORIZON = @isdefined(UPDATE_HORIZON) ? UPDATE_HORIZON : 10
MIN_REPLAY_HISTORY = @isdefined(MIN_REPLAY_HISTORY) ? MIN_REPLAY_HISTORY : 200
UPDATE_FREQ = @isdefined(UPDATE_FREQ) ? UPDATE_FREQ : 1
TARGET_UPDATE_FREQ = @isdefined(TARGET_UPDATE_FREQ) ? TARGET_UPDATE_FREQ : 1
DECAY_STEPS = @isdefined(DECAY_STEPS) ? DECAY_STEPS : 2000
CAPACITY = @isdefined(CAPACITY) ? CAPACITY : 2000
CONV_SIZE = @isdefined(CONV_SIZE) ? CONV_SIZE : 8
DENSE_SIZE = @isdefined(DENSE_SIZE) ? DENSE_SIZE : 8

################################################################################
# Agent default default
################################################################################
agent_default_default = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeaturesDefault => CONV_SIZE, Flux.leakyrelu),
                        SeaPearl.GraphConv(CONV_SIZE => CONV_SIZE, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeaturesDefault => CONV_SIZE, Flux.leakyrelu),
                        SeaPearl.GraphConv(CONV_SIZE => CONV_SIZE, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=BATCH_SIZE, #32,
            update_horizon=UPDATE_HORIZON, #what if the number of nodes in a episode is smaller
            min_replay_history=MIN_REPLAY_HISTORY,
            update_freq=UPDATE_FREQ,
            target_update_freq=TARGET_UPDATE_FREQ,
            #rng = rng,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.01,
            #kind = :exp,
            decay_steps=DECAY_STEPS,
            step=1,
            #rng = rng
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=CAPACITY,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (coloring_generator.n,),
    )
)

################################################################################
# Agent Default Chosen
################################################################################
agent_default_chosen = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeaturesDefaultChosen => CONV_SIZE, Flux.leakyrelu),
                        SeaPearl.GraphConv(CONV_SIZE => CONV_SIZE, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.CPNN(
                    graphChain=Flux.Chain(
                        SeaPearl.GraphConv(numInFeaturesDefaultChosen => CONV_SIZE, Flux.leakyrelu),
                        SeaPearl.GraphConv(CONV_SIZE => CONV_SIZE, Flux.leakyrelu),
                    ),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=BATCH_SIZE, #32,
            update_horizon=UPDATE_HORIZON, #what if the number of nodes in a episode is smaller
            min_replay_history=MIN_REPLAY_HISTORY,
            update_freq=UPDATE_FREQ,
            target_update_freq=TARGET_UPDATE_FREQ,
            #rng = rng,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.01,
            #kind = :exp,
            decay_steps=DECAY_STEPS,
            step=1,
            #rng = rng
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=CAPACITY,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (coloring_generator.n,),
    )
)

################################################################################
# Agent Heterogeneous
################################################################################

struct HeterogeneousModel
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
    layer4::SeaPearl.HeterogeneousGraphConv
end

function HeterogeneousModel()
    layer1 = SeaPearl.HeterogeneousGraphConvInit(numInFeaturesHeterogeneous, CONV_SIZE, Flux.leakyrelu)
    layer2 = SeaPearl.HeterogeneousGraphConv(CONV_SIZE => CONV_SIZE, numInFeaturesHeterogeneous, Flux.leakyrelu)
    layer3 = SeaPearl.HeterogeneousGraphConv(CONV_SIZE => CONV_SIZE, numInFeaturesHeterogeneous, Flux.leakyrelu)
    layer4 = SeaPearl.HeterogeneousGraphConv(CONV_SIZE => CONV_SIZE, numInFeaturesHeterogeneous, Flux.leakyrelu)
    return HeterogeneousModel(layer1, layer2, layer3, layer4)
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

agent_heterogeneous = RL.Agent(
    policy=RL.QBasedPolicy(
        learner=RL.DQNLearner(
            approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            target_approximator=RL.NeuralNetworkApproximator(
                model=SeaPearl.HeterogeneousCPNN(
                    graphChain=HeterogeneousModel(),
                    nodeChain=Flux.Chain(
                        Flux.Dense(CONV_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                    ),
                    globalChain=Flux.Chain(),
                    outputChain=Flux.Chain(
                        Flux.Dense(DENSE_SIZE, DENSE_SIZE, Flux.leakyrelu),
                        Flux.Dense(DENSE_SIZE, coloring_generator.n),
                    )
                ),
                optimizer=ADAM()
            ),
            loss_func=Flux.Losses.huber_loss,
            batch_size=BATCH_SIZE, #32,
            update_horizon=UPDATE_HORIZON, #what if the number of nodes in a episode is smaller
            min_replay_history=MIN_REPLAY_HISTORY,
            update_freq=UPDATE_FREQ,
            target_update_freq=TARGET_UPDATE_FREQ,
            #rng = rng,
        ),
        explorer=RL.EpsilonGreedyExplorer(
            ϵ_stable=0.01,
            #kind = :exp,
            decay_steps=DECAY_STEPS,
            step=1,
            #rng = rng
        )
    ),
    trajectory=RL.CircularArraySLARTTrajectory(
        capacity=CAPACITY,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (coloring_generator.n,),
    )
)
