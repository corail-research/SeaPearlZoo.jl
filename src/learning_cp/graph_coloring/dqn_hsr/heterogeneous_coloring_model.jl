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
            Inputlayer = SeaPearl.HeterogeneousGraphConvInit(original_features_size, out, Flux.leakyrelu, init = init)
        else
            Inputlayer = SeaPearl.HeterogeneousGraphConvInit(original_features_size, mid, Flux.leakyrelu, init = init)
            for i in 1:n_layers - 2
                push!(Middlelayers, SeaPearl.HeterogeneousGraphConv(mid => mid, original_features_size, Flux.leakyrelu; init = init))
            end
            push!(Middlelayers, SeaPearl.HeterogeneousGraphConv(mid => out, original_features_size, Flux.leakyrelu; init = init))
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

function build_graph_coloring_approximator_model(device)
    approximator_model = SeaPearl.HeterogeneousFullFeaturedCPNN(
        graphChain = Flux.Chain(HeterogeneousModel([6,5,2], 8, 8, 3)),
        varChain = Flux.Chain(
            Flux.Dense(8, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 16, init=Flux.glorot_uniform(MersenneTwister(42)))
            ),
        valChain = Flux.Chain(
            Flux.Dense(8, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 16, init=Flux.glorot_uniform(MersenneTwister(42)))
            ),
        globalChain = Flux.Chain(),
        outputChain = Flux.Chain(
            Flux.Dense(32, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 1, init=Flux.glorot_uniform(MersenneTwister(42))),
            )
    )|> device
    return approximator_model
end


function build_graph_coloring_target_approximator_model(device)
    target_approximator_model = SeaPearl.HeterogeneousFullFeaturedCPNN(
        graphChain = Flux.Chain(HeterogeneousModel([6,5,2], 8, 8, 3)),
        varChain = Flux.Chain(
            Flux.Dense(8, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 16, init=Flux.glorot_uniform(MersenneTwister(42)))),
        valChain = Flux.Chain(
            Flux.Dense(8, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 16, init=Flux.glorot_uniform(MersenneTwister(42)))),
        globalChain = Flux.Chain(),
        outputChain = Flux.Chain(
            Flux.Dense(32, 16, Flux.leakyrelu, init=Flux.glorot_uniform(MersenneTwister(42))),
            Flux.Dense(16, 1, init=Flux.glorot_uniform(MersenneTwister(42))),
            )
    )|> device
    return target_approximator_model
end

function build_graph_coloring_agent(approximator_model, target_approximator_model, agent_config::ColoringAgentConfig, rng, decay_steps)
    agent = RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator=RL.NeuralNetworkApproximator(
                    model=approximator_model,
                    optimizer=ADAM()
                ),
                target_approximator=RL.NeuralNetworkApproximator(
                    model=target_approximator_model,
                    optimizer=ADAM()
                ),
                loss_func=Flux.Losses.huber_loss,
                batch_size=agent_config.batch_size,
                update_horizon=agent_config.update_horizon,
                min_replay_history=agent_config.min_replay_history,
                update_freq=agent_config.update_freq,
                target_update_freq=agent_config.target_update_freq,
                γ = 0.99f0
            ),
            explorer = get_epsilon_greedy_explorer(decay_steps, 0.05; rng)
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity=agent_config.trajectory_capacity,
            state=SeaPearl.HeterogeneousTrajectoryState[] => (),
            legal_actions_mask=Vector{Bool} => (agent_config.output_size,),
        )
    )
    return agent
end