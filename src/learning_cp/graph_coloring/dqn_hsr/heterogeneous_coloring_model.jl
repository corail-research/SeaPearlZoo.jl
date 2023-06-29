# Model definition

struct HeterogeneousModel{A,B}
    Inputlayer::A
    Middlelayers::Vector{B}

    function HeterogeneousModel(Inputlayer, Middlelayers)
        return new{typeof(Inputlayer), eltype(Middlelayers)}(Inputlayer,Middlelayers)
    end

    function HeterogeneousModel(original_features_size::Vector{Int}, mid::Int, out::Int, n_layers::Int; init = Flux.glorot_uniform)
        pool = SeaPearl.sumPooling()
        Middlelayers=SeaPearl.HeterogeneousGraphConv[]

        if n_layers == 1 
            Inputlayer = get_heterogeneous_graph_conv_init_layer(original_features_size, out, init = init)
        else
            Inputlayer = get_heterogeneous_graph_conv_init_layer(original_features_size, mid, init = init)
            for i in 1:n_layers - 2
                push!(Middlelayers, get_heterogeneous_graph_conv_layer(mid, mid, original_features_size, init = init))
            end
            push!(Middlelayers,get_heterogeneous_graph_conv_layer(mid, out, original_features_size, init = init))
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

get_heterogeneous_graph_conv_layer(in, out, original_features_size; init = Flux.glorot_uniform) = SeaPearl.HeterogeneousGraphConv(in => out, original_features_size, Flux.leakyrelu; init = init)

get_heterogeneous_graph_conv_init_layer(original_features_size, out; init = Flux.glorot_uniform) = SeaPearl.HeterogeneousGraphConvInit(original_features_size, out, Flux.leakyrelu, init = init)

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

function get_heterogeneous_agent(; get_explorer, batch_size=16, update_horizon, min_replay_history, update_freq=1, target_update_freq=200, γ = 0.999f0, get_heterogeneous_trajectory, get_heterogeneous_nn)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_heterogeneous_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, get_heterogeneous_nn, γ),
            explorer=get_explorer(),
        ),
        trajectory=get_heterogeneous_trajectory()
    )
end

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

function get_heterogeneous_graph_chain(original_features_size, mid, out, n_layers; init = Flux.glorot_uniform, device = cpu)
    @assert n_layers >= 1
    return HeterogeneousModel(original_features_size, mid, out, n_layers; init = init)
end

function get_heterogeneous_fullfeaturedcpnn(;feature_size, conv_type="gc", conv_size=8, dense_size=16, output_size=1, n_layers_graph=3, n_layers_node=2, n_layers_output=2, σ=Flux.leakyrelu, heads=4, init = Flux.glorot_uniform, device = cpu)
    if conv_type == "gc"
        return SeaPearl.HeterogeneousFullFeaturedCPNN(
            get_heterogeneous_graph_chain(feature_size, conv_size, conv_size, n_layers_graph; init = init),
            get_dense_chain(conv_size, dense_size, dense_size, n_layers_node, σ, init = init),
            Flux.Chain(),
            get_dense_chain(2*dense_size, dense_size, output_size, n_layers_output, σ, init = init)
            )|> device
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
