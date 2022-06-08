using SeaPearl

###############################################################################
######### Utils
###############################################################################

get_default_graph_conv_layer(in, out) = SeaPearl.GraphConv(in => out, Flux.leakyrelu)

function get_default_graph_chain(in, mid, out, n_layers)
    @assert n_layers >= 1
    layers = []
    if n_layers == 1
        push!(layers, get_default_graph_conv_layer(in, out))
    else
        push!(layers, get_default_graph_conv_layer(in, mid))
        for i in 2:n_layers
            push!(layers, get_default_graph_conv_layer(mid, out))
        end
    end
    return Flux.Chain(layers...)
end

function get_dense_chain(in, mid, out, n_layers)
    @assert n_layers >= 1
    layers = []
    if n_layers == 1
        push!(layers, Flux.Dense(in, out))
    else
        push!(layers, Flux.Dense(in, mid))
        for i in 2:n_layers
            push!(layers, Flux.Dense(mid, out))
        end
    end
    return Flux.Chain(layers...)
end

function get_default_cpnn(;feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output)
    return SeaPearl.CPNN(
        graphChain=get_default_graph_chain(feature_size, conv_size, conv_size, n_layers_graph),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
        outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output)
    )
end

function get_default_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_default_nn)
    return RL.DQNLearner(
        approximator=RL.NeuralNetworkApproximator(
            model=get_default_nn(),
            optimizer=ADAM()
        ),
        target_approximator=RL.NeuralNetworkApproximator(
            model=get_default_nn(),
            optimizer=ADAM()
        ),
        loss_func=Flux.Losses.huber_loss,
        batch_size=batch_size,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_freq=update_freq,
        target_update_freq=target_update_freq,
    )
end

function get_epsilon_greedy_explorer(decay_steps, ϵ_stable)
    return RL.EpsilonGreedyExplorer(
        ϵ_stable=ϵ_stable,
        kind=:exp,
        decay_steps=decay_steps,
        step=1,
    )
end

function get_ucb_explorer(c, n_actions)
    return RL.UCBExplorer(n_actions; c=c)
end

function get_default_slart_trajectory(; capacity, n_actions)
    return RL.CircularArraySLARTTrajectory(
        capacity=capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_actions,),
    )
end

function get_default_sart_trajectory(; capacity)
    return RL.CircularArraySARTTrajectory(
        capacity=capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
    )
end

function get_default_agent(;get_explorer, batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_default_nn, get_default_trajectory)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_default_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_default_nn),
            explorer=get_explorer(),
        ),
        trajectory=get_default_trajectory()
    )
end

struct HeterogeneousModel1
    layer1::SeaPearl.HeterogeneousGraphConvInit
end

function (m::HeterogeneousModel1)(fg)
    out1 = m.layer1(fg)
    return out1
end

Flux.@functor HeterogeneousModel1

struct HeterogeneousModel2
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
end

function (m::HeterogeneousModel2)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    return out2
end

Flux.@functor HeterogeneousModel2

struct HeterogeneousModel3
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
end

function (m::HeterogeneousModel3)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    return out3
end

Flux.@functor HeterogeneousModel3

struct HeterogeneousModel4
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
    layer4::SeaPearl.HeterogeneousGraphConv
end

function (m::HeterogeneousModel4)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    out4 = m.layer4(out3, original_fg)
    return out4
end

Flux.@functor HeterogeneousModel4

struct HeterogeneousModel5
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
    layer4::SeaPearl.HeterogeneousGraphConv
    layer5::SeaPearl.HeterogeneousGraphConv
end

function (m::HeterogeneousModel5)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    out4 = m.layer4(out3, original_fg)
    out5 = m.layer5(out4, original_fg)
    return out5
end

Flux.@functor HeterogeneousModel5

struct HeterogeneousModel6
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphConv
    layer3::SeaPearl.HeterogeneousGraphConv
    layer4::SeaPearl.HeterogeneousGraphConv
    layer5::SeaPearl.HeterogeneousGraphConv
    layer6::SeaPearl.HeterogeneousGraphConv
end

function (m::HeterogeneousModel6)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1, original_fg)
    out3 = m.layer3(out2, original_fg)
    out4 = m.layer4(out3, original_fg)
    out5 = m.layer5(out4, original_fg)
    out6 = m.layer6(out5, original_fg)
    return out6
end

Flux.@functor HeterogeneousModel6

get_heterogeneous_graph_conv_layer(in, out, original_features_size, pool) = SeaPearl.HeterogeneousGraphConv(in => out, original_features_size, Flux.leakyrelu; pool=pool)

get_heterogeneous_graph_conv_init_layer(original_features_size, out) = SeaPearl.HeterogeneousGraphConvInit(original_features_size, out, Flux.leakyrelu)

function get_heterogeneous_graph_chain(original_features_size, mid, out, n_layers; pool=SeaPearl.sumPooling())
    @assert n_layers >= 1 and n_layers <= 6
    if n_layers == 1
        return HeterogeneousModel1(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid)
        )
    elseif n_layers == 2
        return HeterogeneousModel2(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool)
        )
    elseif n_layers == 3
        return HeterogeneousModel3(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool),
        )
    elseif n_layers == 4
        return HeterogeneousModel4(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool),
        )
    elseif n_layers == 5
        return HeterogeneousModel5(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool)
        )
    elseif n_layers == 6
        return HeterogeneousModel6(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool),
        )
    end
end

# struct Heterog, pooleneousModelHGT1
#     layer1::SeaPea, poolrl.HeterogeneousGraphConvInit
# end

# function (m::HeterogeneousModelHGT1)(fg)
#     out1 = m.layer1(fg)
#     return ou, poolt1
# e, poolnd

# Flux.@functor HeterogeneousModelHGT1

# struct HeterogeneousModelHGT2
#     layer1::SeaPearl.HeterogeneousGraphConvInit
#     layer2::SeaPearl.HeterogeneousGraphTransformer
# end

# function (m::HeterogeneousModelHGT2)(fg)
#     original_fg = fg
#     out1 = m.layer1(fg)
#     out2 = m.layer2(out1)
#     return out2
# end

# Flux.@functor HeterogeneousModelHGT2

# struct HeterogeneousModelHGT3
#     layer1::SeaPearl.HeterogeneousGraphConvInit
#     layer2::SeaPearl.HeterogeneousGraphTransformer
#     layer3::SeaPearl.HeterogeneousGraphTransformer
# end

# function (m::HeterogeneousModelHGT3)(fg)
#     original_fg = fg
#     out1 = m.layer1(fg)
#     out2 = m.layer2(out1)
#     out3 = m.layer3(out2)
#     return out3
# end

# Flux.@functor HeterogeneousModelHGT3

# struct HeterogeneousModelHGT4
#     layer1::SeaPearl.HeterogeneousGraphConvInit
#     layer2::SeaPearl.HeterogeneousGraphTransformer
#     layer3::SeaPearl.HeterogeneousGraphTransformer
#     layer4::SeaPearl.HeterogeneousGraphTransformer
# end

# function (m::HeterogeneousModelHGT4)(fg)
#     original_fg = fg
#     out1 = m.layer1(fg)
#     out2 = m.layer2(out1)
#     out3 = m.layer3(out2)
#     out4 = m.layer4(out3)
#     return out4
# end

# Flux.@functor HeterogeneousModelHGT4

# struct HeterogeneousModelHGT5
#     layer1::SeaPearl.HeterogeneousGraphConvInit
#     layer2::SeaPearl.HeterogeneousGraphTransformer
#     layer3::SeaPearl.HeterogeneousGraphTransformer
#     layer4::SeaPearl.HeterogeneousGraphTransformer
#     layer5::SeaPearl.HeterogeneousGraphTransformer
# end

# function (m::HeterogeneousModelHGT5)(fg)
#     original_fg = fg
#     out1 = m.layer1(fg)
#     out2 = m.layer2(out1)
#     out3 = m.layer3(out2)
#     out4 = m.layer4(out3)
#     out5 = m.layer5(out4)
#     return out5
# end

# Flux.@functor HeterogeneousModelHGT5

# struct HeterogeneousModelHGT6
#     layer1::SeaPearl.HeterogeneousGraphConvInit
#     layer2::SeaPearl.HeterogeneousGraphTransformer
#     layer3::SeaPearl.HeterogeneousGraphTransformer
#     layer4::SeaPearl.HeterogeneousGraphTransformer
#     layer5::SeaPearl.HeterogeneousGraphTransformer
#     layer6::SeaPearl.HeterogeneousGraphTransformer
# end

# function (m::HeterogeneousModelHGT6)(fg)
#     original_fg = fg
#     out1 = m.layer1(fg)
#     out2 = m.layer2(out1)
#     out3 = m.layer3(out2)
#     out4 = m.layer4(out3)
#     out5 = m.layer5(out4)
#     out6 = m.layer6(out5)
#     return out6
# end

# Flux.@functor HeterogeneousModelHGT6

# get_heterogeneous_graph_conv_layer_hgt(in, heads) = SeaPearl.HeterogeneousGraphTransformer(in, heads)

# function get_heterogeneous_graph_chain_hgt(original_features_size, out, n_layers, heads)
#     @assert n_layers >= 1 and n_layers <= 6
#     if n_layers == 1
#         return HeterogeneousModelHGT1(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out)
#         )
#     elseif n_layers == 2
#         return HeterogeneousModelHGT2(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads)
#         )
#     elseif n_layers == 3
#         return HeterogeneousModelHGT3(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#         )
#     elseif n_layers == 4
#         return HeterogeneousModelHGT4(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#         )
#     elseif n_layers == 5
#         return HeterogeneousModelHGT5(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads)
#         )
#     elseif n_layers == 6
#         return HeterogeneousModelHGT6(
#             get_heterogeneous_graph_conv_init_layer(original_features_size, out),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads),
#             get_heterogeneous_graph_conv_layer_hgt(out, heads)
#         )
#     end
# end

function get_heterogeneous_cpnn(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.sumPooling())
    return SeaPearl.HeterogeneousCPNN(
        graphChain=get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool=pool),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
        outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output)
    )
end

function get_heterogeneous_fullfeaturedcpnn(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.sumPooling())
    return SeaPearl.HeterogeneousFullFeaturedCPNN(
        get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool=pool),
        get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
        Flux.Chain(),
        get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output)
    )
end

function get_heterogeneous_variableoutputcpnn(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.sumPooling())
    return SeaPearl.HeterogeneousVariableOutputCPNN(
        get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool=pool),
        get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
        get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output)
    )
end

# function get_heterogeneous_cpnn_hgt(;feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output, heads)
#     return SeaPearl.HeterogeneousCPNN(
#         graphChain=get_heterogeneous_graph_chain_hgt(feature_size, conv_size, n_layers_graph, heads),
#         nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
#         outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output)
#     )
# end

function get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn)
    return RL.DQNLearner(
        approximator=RL.NeuralNetworkApproximator(
            model=get_heterogeneous_nn(),
            optimizer=ADAM()
        ),
        target_approximator=RL.NeuralNetworkApproximator(
            model=get_heterogeneous_nn(),
            optimizer=ADAM()
        ),
        loss_func=Flux.Losses.huber_loss,
        batch_size=batch_size,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_freq=update_freq,
        target_update_freq=target_update_freq,
    )
end

function get_heterogeneous_slart_trajectory(; capacity, n_actions)
    return RL.CircularArraySLARTTrajectory(
        capacity=capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_actions,),
    )
end

function get_heterogeneous_sart_trajectory(; capacity)
    return RL.CircularArraySARTTrajectory(
        capacity=capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
    )
end

function get_heterogeneous_agent(; get_explorer, batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_trajectory, get_heterogeneous_nn)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn),
            explorer=get_explorer(),
        ),
        trajectory=get_heterogeneous_trajectory()
    )
end