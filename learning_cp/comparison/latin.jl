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


    experiment_representation(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_sizes = [3, 9, [2, 3, 1]], 
        output_size = board_size, 
        generator = latin_generator, 
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


function experiment_heterogeneous_n_conv_latin(board_size, density, n_episodes, n_instances; n_eval=10)
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
        generator=latin_generator,
        SR=SR_heterogeneous,
        chosen_features=chosen_features,
        feature_size=[1, 2, board_size],
        type="heterogeneous", 
        output_size = board_size)
end

function experiment_default_chosen_n_conv_latin(board_size, density,  n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the default representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )

    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=latin_generator,
        SR=SR_default,
        chosen_features=chosen_features,
        feature_size= 6 + board_size,
        type="default_chosen",
        output_size = board_size)
end

function experiment_default_default_n_conv_latin(board_size, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the default representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}


    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=latin_generator,
        SR=SR_default,
        feature_size=3,
        chosen_features=nothing,
        type="default_default",
        output_size = board_size)
end

###############################################################################
######### Experiment Type 3
######### 
######### 
###############################################################################


function experiment_chosen_features_heterogeneous_latin(board_size, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)


    chosen_features_list = [
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 2, board_size]
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
            [2, 2, board_size]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 3, board_size]
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

    experiment_chosen_features_heterogeneous(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=latin_generator,
        chosen_features_list=chosen_features_list,
        type="latin",
        output_size = board_size
        )
end

###############################################################################
######### Experiment Type 4
#########  
######### 
###############################################################################


function experiment_heuristic_heterogeneous_latin(board_size, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compares the simple and the supervised learned heuristic for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
        basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_heuristic_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = board_size, 
        generator = latin_generator, 
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "latin",
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

function experiment_explorer_heterogeneous_latin(board_size, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compares different action explorers for the heterogeneous representation.
        - an agent with the epsilon_greedy explorer 
        - an agent with the upper confident bound explorer
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
        basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_explorer_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = board_size, 
        generator = latin_generator,
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "latin",
        decay_steps=2000,
        c=2.0,
    )
end

###############################################################################
######### Experiment Type 6
#########  
######### 
###############################################################################

function experiment_nn_heterogeneous_latin(board_size, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compares different CPNNs for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :boardSize => board_size,
            :density => density,
        ),
    )

    chosen_features = Dict(
        "constraint_activity" => true,
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "variable_domain_size" => true,
        "values_raw" => true,
    )

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_nn_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features = chosen_features,
        feature_size = [2, 3, 1], 
        output_size = board_size, 
        generator = latin_generator, 
        expParameters = expParameters, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "latin",
        decay_steps=2000,
        c=2.0,
        basicHeuristics=basicHeuristics
    )
end

###############################################################################
######### Experiment Type 8
#########  
######### 
###############################################################################


function experiment_chosen_features_hetcpnn_latin(chosen_features_list, board_size, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)
    restartPerInstances = 1


    experiment_chosen_features_hetcpnn(
        board_size+1,
        board_size,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = board_size, 
        generator=latin_generator,
        chosen_features_list=chosen_features_list, 
        type="latin_"*string(board_size),
        )
end

function experiment_chosen_features_hetffcpnn_latin(chosen_features_list, board_size, density, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    latin_generator = SeaPearl.LatinGenerator(board_size, density)
    restartPerInstances = 1


    experiment_chosen_features_hetffcpnn(
        board_size+1,
        board_size,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = board_size, 
        generator=latin_generator,
        chosen_features_list=chosen_features_list, 
        type="latin_"*string(board_size), 
        )
end


