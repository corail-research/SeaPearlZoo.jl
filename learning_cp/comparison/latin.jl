include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################

function experiment_representation_latin(board_size, density, n_episodes, n_instances; n_layers_graph=2, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """

    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    expParameters = Dict(
        :generatorParameters => Dict(
            :boardSize => board_size,
            :density => density,
        ),
    )

    experiment_representation(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_sizes = [3, 12, [2, 6, 1]], 
        output_size = board_size, 
        generator = latin_generator, 
        expParameters = expParameters, 
        basicHeuristics=nothing, 
        n_layers_graph=n_layers_graph, 
        n_eval=n_eval, 
        reward=reward, 
        type="latin", 
    )
end 

###############################################################################
######### Experiment Type 2
#########  
######### 
###############################################################################


function experiment_heterogeneous_n_conv(board_size, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )

    experiment_n_conv(board_size, n_episodes, n_instances;
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

    experiment_n_conv(board_size, n_episodes, n_instances;
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

    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=coloring_generator,
        SR=SR_default,
        feature_size=3,
        chosen_features=nothing,
        type="default_default")
end

###############################################################################
######### Experiment Type 3
######### 
######### 
###############################################################################


###############################################################################
######### Experiment Type 4
#########  
######### 
###############################################################################