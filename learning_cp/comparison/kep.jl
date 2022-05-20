include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

function experiment_representation(n_nodes, density, n_episodes, n_instances; n_layers_graph=2, n_eval=10)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    kep_generator = SeaPearl.KepGenerator(n_nodes, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

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
        output_size=2,
        n_layers_graph=n_layers_graph,
        n_layers_node=2,
        n_layers_output=2
    )
    learned_heuristic_default_default = SeaPearl.SimpleLearnedHeuristic{SR_default,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_default)

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_raw" => true,
    )

    agent_default_chosen = get_default_agent(;
        capacity=2000,
        decay_steps=2000,
        ϵ_stable=0.01,
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        feature_size=11,
        conv_size=8,
        dense_size=16,
        output_size=2,
        n_layers_graph=n_layers_graph,
        n_layers_node=2,
        n_layers_output=2
    )
    learned_heuristic_default_chosen = SeaPearl.SimpleLearnedHeuristic{SR_default,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_chosen; chosen_features=chosen_features)

    agent_heterogeneous = get_heterogeneous_agent(;
        capacity=2000,
        decay_steps=2000,
        ϵ_stable=0.01,
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        feature_size=[1, 6, 1],
        conv_size=8,
        dense_size=16,
        output_size=2,
        n_layers_graph=n_layers_graph,
        n_layers_node=2,
        n_layers_output=2
    )
    learned_heuristic_heterogeneous = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_heterogeneous; chosen_features=chosen_features)


    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)

    learnedHeuristics = OrderedDict(
        # "defaultdefault" => learned_heuristic_default_default,
        "defaultchosen" => learned_heuristic_default_chosen,
        # "heterogeneous" => learned_heuristic_heterogeneous,
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
            :density => density
        ),
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=1,
        generator=kep_generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=false,
        expParameters=expParameters,
        nbRandomHeuristics=0,
        exp_name="kep_representation_" * string(n_episodes) * "_" * string(n_nodes) * "_"
    )
end

experiment_representation(2, 0.8, 10, 1)
nothing