using SeaPearl

###############################################################################
######### Utils
###############################################################################

get_default_graph_conv_layer(in, out, pool; init = Flux.glorot_uniform) = SeaPearl.GraphConv(in => out, Flux.leakyrelu; pool = pool, init = init)

function get_default_graph_chain(in, mid, out, n_layers,pool; init = Flux.glorot_uniform )
    @assert n_layers >= 1
    layers = []
    if n_layers == 1
        push!(layers, get_default_graph_conv_layer(in, out, pool; init = init))
    elseif n_layers == 2
        push!(layers, get_default_graph_conv_layer(in, mid,pool; init = init))
        push!(layers, get_default_graph_conv_layer(mid, out, pool; init = init))
    else
        push!(layers, get_default_graph_conv_layer(in, mid, pool; init = init))
        for i in 2:(n_layers-1)
            push!(layers, get_default_graph_conv_layer(mid, mid, pool; init = init))
        end
        push!(layers, get_default_graph_conv_layer(mid, out, pool; init = init))
    end
    return Flux.Chain(layers...)
end


function get_dense_chain(in, mid, out, n_layers, σ=Flux.identity; init = Flux.glorot_uniform )
    @assert n_layers >= 1
    layers = []
    if n_layers == 1
        push!(layers, Flux.Dense(in, out, init= init))
    elseif n_layers == 2
        push!(layers, Flux.Dense(in, mid, σ, init= init))
        push!(layers, Flux.Dense(mid, out, init= init))
    else
        push!(layers, Flux.Dense(in, mid, σ, init= init))
        for i in 2:(n_layers-1)
            push!(layers, Flux.Dense(mid, mid, σ, init= init))
        end
        push!(layers, Flux.Dense(mid, out, init= init))
    end
    return Flux.Chain(layers...)
end

function get_default_cpnn(;feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output, pool=SeaPearl.sumPooling(), σ=Flux.leakyrelu, init = Flux.glorot_uniform)
    return SeaPearl.CPNN(
        graphChain=get_default_graph_chain(feature_size, conv_size, conv_size, n_layers_graph, pool; init = init),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ; init=init),
        outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output, σ; init=init)
    )
end

function get_default_ffcpnn(;feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output, pool=SeaPearl.meanPooling(), σ=Flux.leakyrelu, init = Flux.glorot_uniform)
    return SeaPearl.FullFeaturedCPNN(
        graphChain=get_default_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool = pool, init = init),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ; init=init),
        outputChain=get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output, σ; init=init)
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

function get_epsilon_greedy_explorer(decay_steps, ϵ_stable; rng=nothing)
    if isnothing(rng)
        return RL.EpsilonGreedyExplorer(
            ϵ_stable=ϵ_stable,
            kind=:exp,
            decay_steps=decay_steps,
            step=1
        )
    else
        return RL.EpsilonGreedyExplorer(
            ϵ_stable=ϵ_stable,
            kind=:exp,
            decay_steps=decay_steps,
            step=1,
            rng = rng
        )
    end
end

function get_ucb_explorer(c, n_actions)
    return RL.UCBExplorer(n_actions; c=c)
end

function get_softmax_explorer(T_stable, T_init, decay_steps)
    return SeaPearl.SoftmaxTDecayExplorer(;T_init=T_init, T_stable=T_stable, decay_steps=decay_steps)
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


struct HeterogeneousModel{A,B}
    Inputlayer::A
    Middlelayers::Vector{B}

    function HeterogeneousModel(Inputlayer, Middlelayers)
        return new{typeof(Inputlayer), eltype(Middlelayers)}(Inputlayer,Middlelayers)
    end

    function HeterogeneousModel(original_features_size::Vector{Int}, mid::Int, out::Int, n_layers::Int; pool=SeaPearl.meanPooling(), init = Flux.glorot_uniform)
    
    Middlelayers=SeaPearl.HeterogeneousGraphConv[]

    if n_layers == 1 
        Inputlayer = get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init)
    else
        Inputlayer = get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init)
        for i in 1:n_layers - 2
            push!(Middlelayers, get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init))
        end
        push!(Middlelayers,get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init))
    end
    return new{typeof(Inputlayer), eltype(Middlelayers)}(Inputlayer, Middlelayers)

    end
end

Flux.@functor HeterogeneousModel
"""
function Flux.functor(::Type{<:HeterogeneousModel}, m)
    return (m.Inputlayer, m.Middlelayers), ls -> HeterogeneousModel(ls[1], ls[2])
end
"""
function (m::HeterogeneousModel)(fg)
    original_fg = deepcopy(fg)
    out = m.Inputlayer(fg)
    for layer in m.Middlelayers
        out = layer(out, original_fg)
    end
    return out
end



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

#=
struct HGTModel1
    layer1::SeaPearl.HeterogeneousGraphConvInit
end

function (m::HGTModel1)(fg)
    out1 = m.layer1(fg)
    return out1
end

Flux.@functor HGTModel1

struct HGTModel2
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphTransformer
end

function (m::HGTModel2)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1)
    return out2
end

Flux.@functor HGTModel2

struct HGTModel3
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphTransformer
    layer3::SeaPearl.HeterogeneousGraphTransformer
end

function (m::HGTModel3)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1)
    out3 = m.layer3(out2)
    return out3
end

Flux.@functor HGTModel3

struct HGTModel4
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphTransformer
    layer3::SeaPearl.HeterogeneousGraphTransformer
    layer4::SeaPearl.HeterogeneousGraphTransformer
end

function (m::HGTModel4)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1)
    out3 = m.layer3(out2)
    out4 = m.layer4(out3)
    return out4
end

Flux.@functor HGTModel4

struct HGTModel5
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphTransformer
    layer3::SeaPearl.HeterogeneousGraphTransformer
    layer4::SeaPearl.HeterogeneousGraphTransformer
    layer5::SeaPearl.HeterogeneousGraphTransformer
end

function (m::HGTModel5)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1)
    out3 = m.layer3(out2)
    out4 = m.layer4(out3)
    out5 = m.layer5(out4)
    return out5
end

Flux.@functor HGTModel5

struct HGTModel6
    layer1::SeaPearl.HeterogeneousGraphConvInit
    layer2::SeaPearl.HeterogeneousGraphTransformer
    layer3::SeaPearl.HeterogeneousGraphTransformer
    layer4::SeaPearl.HeterogeneousGraphTransformer
    layer5::SeaPearl.HeterogeneousGraphTransformer
    layer6::SeaPearl.HeterogeneousGraphTransformer
end

function (m::HGTModel6)(fg)
    original_fg = fg
    out1 = m.layer1(fg)
    out2 = m.layer2(out1)
    out3 = m.layer3(out2)
    out4 = m.layer4(out3)
    out5 = m.layer5(out4)
    out6 = m.layer6(out5)
    return out6
end

Flux.@functor HGTModel6
=#
get_heterogeneous_graph_conv_layer(in, out, original_features_size, pool; init = Flux.glorot_uniform) = SeaPearl.HeterogeneousGraphConv(in => out, original_features_size, Flux.leakyrelu; pool = pool, init = init)

get_heterogeneous_graph_conv_init_layer(original_features_size, out; init = Flux.glorot_uniform) = SeaPearl.HeterogeneousGraphConvInit(original_features_size, out, Flux.leakyrelu, init = init)

#=function get_heterogeneous_graph_chain(original_features_size, mid, out, n_layers; pool=SeaPearl.meanPooling(), init = Flux.glorot_uniform)
    @assert n_layers >= 1 and n_layers <= 6
    if n_layers == 1
        return HeterogeneousModel1(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init)
        )
    elseif n_layers == 2
        return HeterogeneousModel2(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init)
        )
    elseif n_layers == 3
        return HeterogeneousModel3(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init),
        )
    elseif n_layers == 4
        return HeterogeneousModel4(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init),
        )
    elseif n_layers == 5
        return HeterogeneousModel5(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init)
        )
    elseif n_layers == 6
        return HeterogeneousModel6(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init),
        )
    elseif n_layers == 24
        return HeterogeneousModel24(
            get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, pool, init = init),
            get_heterogeneous_graph_conv_layer(mid, out, original_features_size, pool, init = init),
        )
    end
end
=#

function get_heterogeneous_graph_chain(original_features_size, mid, out, n_layers; pool=SeaPearl.meanPooling(), init = Flux.glorot_uniform, device = cpu)
    @assert n_layers >= 1
    return HeterogeneousModel(original_features_size, mid, out, n_layers; pool= pool, init = init)
end

get_hgt_layer(dim, heads) = SeaPearl.HeterogeneousGraphTransformer(dim, heads)

function get_hgt(original_features_size, out, n_layers; heads=4, init = Flux.glorot_uniform)
    @assert n_layers >= 1 and n_layers <= 6
    if n_layers == 1
        return HGTModel1(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init)
        )
    elseif n_layers == 2
        return HGTModel2(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init),
            get_hgt_layer(out, heads)
        )
    elseif n_layers == 3
        return HGTModel3(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
        )
    elseif n_layers == 4
        return HGTModel4(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
        )
    elseif n_layers == 5
        return HGTModel5(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads)
        )
    elseif n_layers == 6
        return HGTModel6(
            get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
            get_hgt_layer(out, heads),
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

function get_heterogeneous_cpnn(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.meanPooling(), σ=Flux.leakyrelu, init = Flux.glorot_uniform)
    return SeaPearl.HeterogeneousCPNN(
        graphChain=get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool=pool, init = init),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ, init = init),
        outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output, σ, init = init)
    )
end

function get_heterogeneous_fullfeaturedcpnn(;feature_size, conv_type="gc", conv_size=8, dense_size=16, output_size=1, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.meanPooling(), σ=Flux.leakyrelu, heads=4, init = Flux.glorot_uniform)
    if conv_type == "gc"
        return SeaPearl.HeterogeneousFullFeaturedCPNN(
            get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; pool=pool, init = init),
            get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ, init = init),
            Flux.Chain(),
            get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output, σ, init = init)
         )#|> device
    elseif conv_type == "hgt"
        return SeaPearl.HeterogeneousFullFeaturedCPNN(
            get_hgt(feature_size, conv_size, n_layers_graph; heads=heads, init = init),
            get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ, init = init),
            Flux.Chain(),
            get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output, σ, init = init)
        )
    else
        error("conv_type unknown!")
    end
end

function get_pretrained_heterogeneous_fullfeaturedcpnn(file_path)
    @load file_path model
    return deepcopy(model.model)
end

function get_heterogeneous_ffcpnnv2(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_output=2, pool=SeaPearl.meanPooling())
    return SeaPearl.HeterogeneousFFCPNNv2(
        get_heterogeneous_graph_chain(feature_size, conv_size, dense_size, n_layers_graph; pool=pool),
        Flux.Chain(),
        get_dense_chain(5*dense_size, dense_size, output_size, n_layers_output) #TODO: fix the 'in' argument (hardcoded)
    )
end

function get_heterogeneous_ffcpnnv3(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_output=2, pool=SeaPearl.meanPooling(), σ=Flux.leakyrelu, pooling="mean")
    return SeaPearl.HeterogeneousFFCPNNv3(
        get_heterogeneous_graph_chain(feature_size, conv_size, dense_size, n_layers_graph; pool=pool),
        Flux.Chain(),
        get_dense_chain(dense_size, dense_size, output_size, n_layers_output, σ);
        pooling=pooling 
    )
end

function get_heterogeneous_ffcpnnv4(;feature_size, conv_size=8, dense_size=16, output_size, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.meanPooling(), σ=Flux.leakyrelu, pooling="mean")
    return SeaPearl.HeterogeneousFFCPNNv4(
        get_heterogeneous_graph_chain(feature_size, conv_size, dense_size, n_layers_graph; pool=pool),
        get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ),
        Flux.Chain(),
        get_dense_chain(dense_size, dense_size, output_size, n_layers_output, σ);
        pooling=pooling
    )
end

function get_heterogeneous_variableoutputcpnn(;feature_size, conv_size=8, dense_size=16, output_size=1, n_layers_graph=3, n_layers_node=2, n_layers_output=2, pool=SeaPearl.meanPooling())
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

function get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn, γ)
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
        γ = γ
    )
end

function get_heterogeneous_slart_trajectory(; capacity, n_actions)
    return RL.CircularArraySLARTTrajectory(
        capacity=capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_actions,),
    )
end

function get_heterogeneous_ppo_trajectory(; capacity, n_actions)
    return RL.MaskedPPOTrajectory(
        capacity=capacity,
        state=SeaPearl.HeterogeneousTrajectoryState[] => (),
        legal_actions_mask=Matrix{Bool} => (n_actions,1),
        action = Vector{Int} => (1,),
        action_log_prob=Vector{Float32} => (1,),
        reward =  Vector{Float32} => (1,),
        terminal = Vector{Bool} => (1,),
    )
end

CircularArrayPSLARTTrajectory(; capacity, kwargs...) = RL.PrioritizedTrajectory(
    RL.CircularArraySLARTTrajectory(; capacity = capacity, kwargs...),
    RL.SumTree(capacity),
)

function get_heterogeneous_prioritized_trajectory(; capacity, n_actions)
    return CircularArrayPSLARTTrajectory(
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

function get_heterogeneous_agent(; get_explorer, batch_size=16, update_horizon, min_replay_history, update_freq=1, target_update_freq=200, γ = 0.999f0, get_heterogeneous_trajectory, get_heterogeneous_nn)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn, γ),
            explorer=get_explorer(),
        ),
        trajectory=get_heterogeneous_trajectory()
    )
end

function get_heterogeneous_agent_priodqn(; get_explorer, batch_size=16, update_horizon, min_replay_history, update_freq=1, target_update_freq=200, get_heterogeneous_prioritized_trajectory, get_heterogeneous_nn)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn),
            explorer=get_explorer(),
        ),
        trajectory=get_heterogeneous_prioritized_trajectory()
    )
end

function get_ppo_approximator(get_heterogeneous_nn_actor,get_heterogeneous_nn_critic)
    return(
        RL.ActorCritic(
            actor =get_heterogeneous_nn_actor(),
            critic =get_heterogeneous_nn_critic(),
            optimizer = ADAM(),
        )
    )
end

function get_heterogeneous_agent_ppo(; n_epochs, n_microbatches, critic_loss_weight = 1.0f0, entropy_loss_weight = 0.01f0, update_freq, get_heterogeneous_ppo_trajectory, get_heterogeneous_nn_actor,get_heterogeneous_nn_critic)
    return RL.Agent(
        policy=RL.PPOPolicy(
            approximator = get_ppo_approximator(get_heterogeneous_nn_actor,get_heterogeneous_nn_critic),
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = 0.2f0,
            max_grad_norm = 0.5f0,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            update_freq  =  update_freq,
        ),
        trajectory=get_heterogeneous_ppo_trajectory()
    )
end