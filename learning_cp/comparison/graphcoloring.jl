include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################

function experiment_representation_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=2, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
    )

    experiment_representation(n_nodes, n_episodes, n_instances;
        chosen_features=nothing,
        feature_sizes = [3, 9, [2, 3, 1]], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        expParameters = expParameters, 
        basicHeuristics=nothing, 
        n_layers_graph=n_layers_graph, 
        n_eval=n_eval, 
        reward=reward, 
        type="nqueens", 
    )
end

experiment_representation_graphcoloring(10, 5, 0.95, 1001, 10; n_layers_graph=3)

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

    experiment_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_heterogeneous,
        chosen_features=chosen_features,
        feature_size=[1, 2, n_nodes],
        type="heterogeneous")
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

    experiment_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_default,
        chosen_features=chosen_features,
        feature_size=6 + n_nodes,
        type="default_chosen")
end

function experiment_default_default_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the default representation.
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    experiment_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_default,
        feature_size=3,
        chosen_features=nothing,
        type="default_default")
end

# println("start experiment_1")
# experiment_heterogeneous_n_conv(10, 5, 0.95, 1001, 1)
# println("end experiment_1")

# experiment_default_chosen_n_conv(10, 5, 0.95, 1001, 10)
# experiment_default_default_n_conv(10, 5, 0.95, 1001, 10)

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

    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
    )

    experiment_chosen_features_heterogeneous(n_nodes, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        chosen_features_list=chosen_features_list,
        type="graphcoloring",
        expParameters = expParameters,
        output_size = n_nodes
        )
end

# experiment_chosen_features_heterogeneous_graphcoloring(10, 5, 0.95, 1001, 10)