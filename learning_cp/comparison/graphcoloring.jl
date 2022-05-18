include("../common/experiment.jl")

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

function get_default_cpnn(feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output)
    return SeaPearl.CPNN(
        graphChain=get_default_graph_chain(feature_size, conv_size, conv_size, n_layers_graph),
        nodeChain=get_dense_chain(conv_size, dense_size, dense_size, n_layers_node),
        outputChain=get_dense_chain(dense_size, dense_size, output_size, n_layers_output)
    )
end

function get_default_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output)
    return RL.DQNLearner(
        approximator=RL.NeuralNetworkApproximator(
            model=get_default_cpnn(feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output),
            optimizer=ADAM()
        ),
        target_approximator=RL.NeuralNetworkApproximator(
            model=get_default_cpnn(feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output),
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

function get_explorer(decay_steps, ϵ_stable)
    return RL.EpsilonGreedyExplorer(
        ϵ_stable=ϵ_stable,
        kind=:exp,
        decay_steps=decay_steps,
        step=1,
    )
end

function get_default_trajectory(capacity, n_actions)
    return RL.CircularArraySLARTTrajectory(
        capacity=capacity,
        state=SeaPearl.DefaultTrajectoryState[] => (),
        legal_actions_mask=Vector{Bool} => (n_actions,),
    )
end

function get_default_agent(; capacity, decay_steps, ϵ_stable, batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output)
    return RL.Agent(
        policy=RL.QBasedPolicy(
            learner=get_default_learner(batch_size, update_horizon, min_replay_history, update_freq, target_update_freq, feature_size, conv_size, dense_size, output_size, n_layers_graph, n_layers_node, n_layers_output),
            explorer=get_explorer(decay_steps, ϵ_stable),
        ),
        trajectory=get_default_trajectory(capacity, output_size)
    )
end

###############################################################################
######### Experiment 1
#########  
######### 
###############################################################################

function experiment_1(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=2)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    agent_default_default = get_default_agent(;
        capacity=2000,
        decay_steps=2000,
        ϵ_stable=0.01,
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        feature_size=3,
        conv_size=8,
        dense_size=16,
        output_size=n_nodes,
        n_layers_graph=n_layers_graph,
        n_layers_node=2,
        n_layers_output=2
    )
    learned_heuristic_default_default = SeaPearl.SimpleLearnedHeuristic{SR_default,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_default)

    agent_default_chosen = get_default_agent(;
        capacity=2000,
        decay_steps=2000,
        ϵ_stable=0.01,
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        feature_size=3,
        conv_size=8,
        dense_size=16,
        output_size=n_nodes,
        n_layers_graph=n_layers_graph,
        n_layers_node=2,
        n_layers_output=2
    )
    learned_heuristic_default_chosen = SeaPearl.SimpleLearnedHeuristic{SR_default,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_chosen)


    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)

    learnedHeuristics = OrderedDict(
        "defaultdefault" => learned_heuristic_default_default,
        "defaultchosen" => learned_heuristic_default_default,
    )
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    expParameters = Dict(
    :generatorParameters => Dict(
        :nbNodes => n_nodes,
        :nbMinColor => n_min_color,
        :density => density
    ),
)

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes/10)),
        nbInstances=n_instances,
        restartPerInstances=1,
        generator=coloring_generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=false,
        expParameters=expParameters,
        nbRandomHeuristics=0
    )
end

experiment_1(10, 5, 0.95, 101, 1)