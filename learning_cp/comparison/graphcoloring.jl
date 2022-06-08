include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################

function experiment_representation_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)


    experiment_representation(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_sizes = [3, 9, [2, 3, 1]], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        basicHeuristics=nothing, 
        n_layers_graph=n_layers_graph, 
        n_eval=n_eval, 
        reward=reward, 
        type="graphcoloring", 
    )
end

###############################################################################
######### Experiment Type 2
#########  
######### 
###############################################################################

function experiment_heterogeneous_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )

    experiment_n_conv(n_nodes, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_heterogeneous,
        chosen_features=chosen_features,
        feature_size=[1, 2, n_nodes],
        type="heterogeneous",
        output_size = n_nodes)
end

function experiment_default_chosen_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the default representation.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )

    experiment_n_conv(n_nodes, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_default,
        chosen_features=chosen_features,
        feature_size=6 + n_nodes,
        type="default_chosen",
        output_size = n_nodes)
end

function experiment_default_default_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the default representation.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    experiment_n_conv(n_nodes, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_default,
        feature_size=3,
        chosen_features=nothing,
        type="default_default",
        output_size = n_nodes)
end

###############################################################################
######### Experiment Type 3
#########  
######### 
###############################################################################

function experiment_chosen_features_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

    chosen_features_list = [
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 2, n_nodes]
        ],
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_raw" => true,
            ), 
            [1, 2, 1]
        ],
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "variable_domain_size" => true,
                "values_onehot" => true,
            ), 
            [2, 2, n_nodes]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 3, n_nodes]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "variable_domain_size" => true,
                "values_raw" => true,
            ), 
            [2, 3, 1]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "nb_not_bounded_variable" => true,
                "variable_initial_domain_size" => true,
                "variable_domain_size" => true,
                "variable_is_bound" => true,
                "values_raw" => true,
            ), 
            [3, 4, 1]
        ],
    ]


    experiment_chosen_features_heterogeneous(n_nodes, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        chosen_features_list=chosen_features_list,
        type="graphcoloring",
        output_size = n_nodes
        )
end

###############################################################################
######### Experiment Type 4
#########  
######### 
###############################################################################

function experiment_heuristic_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
        basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_heuristic_heterogeneous(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator,  
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        eta_decay_steps = Int(floor(n_episodes/1.5)),
        helpValueHeuristic = heuristic_min,
        eta_init = 1.0,
        eta_stable = 0.0
    )
end

###############################################################################
######### Experiment Type 5
#########  
######### 
###############################################################################

function experiment_explorer_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
        basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_explorer_heterogeneous(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        decay_steps=2000,
        c=2.0
    )
end

###############################################################################
######### Experiment Type 6
#########  
######### 
###############################################################################

function experiment_nn_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_nn_heterogeneous(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        decay_steps=2000,
        c=2.0,
        basicHeuristics=basicHeuristics
    )
end

###############################################################################
######### Experiment Type 7
#########  
######### 
###############################################################################

function experiment_pooling_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_pooling_heterogeneous(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        decay_steps=2000,
        c=2.0
    )
end

###############################################################################
######### Experiment Type 8
#########  
######### 
###############################################################################


function experiment_chosen_features_hetcpnn_graphcoloring(chosen_features_list, n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    restartPerInstances = 1

    experiment_chosen_features_hetcpnn(
        n_nodes,
        n_nodes+1,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = n_nodes, 
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type="graphcoloring_"*string(n_nodes),
        )
end

function experiment_chosen_features_hetffcpnn_graphcoloring(chosen_features_list, n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    restartPerInstances = 1

    experiment_chosen_features_hetffcpnn(
        n_nodes,
        n_nodes+1,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = n_nodes, 
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type="graphcoloring_"*string(n_nodes)
        )
end