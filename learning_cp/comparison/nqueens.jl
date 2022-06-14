include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################

"""
Compares three agents:
    - an agent with the default graph representation and default features;
    - an agent with the default graph representation and chosen features;
    - an agent with the heterogeneous graph representation and chosen features.
"""
function experiment_representation_nqueens(board_size, n_episodes, n_instances; n_layers_graph=2, n_eval=10, reward=SeaPearl.GeneralReward)
    
    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    experiment_representation(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_sizes = [3, 12, [2, 6, 1]], 
        output_size = board_size, 
        generator = nqueens_generator, 
        basicHeuristics=nothing, 
        n_layers_graph=n_layers_graph, 
        n_eval=n_eval, 
        reward=reward, 
        type="nqueens", 
    )
end

###############################################################################
######### Experiment Type 2
#########  
######### 
###############################################################################

"""
Compares the impact of the number of convolution layers for the heterogeneous representation.
"""
function experiment_heterogeneous_n_conv(board_size, n_episodes, n_instances; n_eval=10)
   
    nqueens_generator = SeaPearl.ClusterizedGraphColoringGenerator(board_size)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )

    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=nqueens_generator,
        SR=SR_heterogeneous,
        chosen_features=chosen_features,
        feature_size=[1, 2, board_size],
        type="heterogeneous",
        output_size = board_size)
end

"""
Compares the impact of the number of convolution layers for the default representation.
"""
function experiment_default_chosen_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    
    nqueens_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}

    chosen_features = Dict(
        "constraint_type" => true,
        "variable_initial_domain_size" => true,
        "values_onehot" => true,
    )



    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=nqueens_generator,
        SR=SR_default,
        chosen_features=chosen_features,
        feature_size=6 + n_nodes,
        type="default_chosen",
        output_size = board_size)
end

"""
Compares the impact of the number of convolution layers for the default representation.
"""
function experiment_default_default_n_conv(board_size, n_episodes, n_instances; n_eval=10)
    
    nqueens_generator = SeaPearl.ClusterizedGraphColoringGenerator(board_size)
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}


    experiment_n_conv(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=nqueens_generator,
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
"""
Compares the impact of the chosen features for the heterogeneous representation.
"""
function experiment_chosen_features_heterogeneous_nqueens(board_size, n_episodes, n_instances; n_eval=10, reward=SeaPearl.GeneralReward)
    
    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    chosen_features_list = [
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 5, board_size]
        ],
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_raw" => true,
            ), 
            [1, 5, 1]
        ],
        [
            Dict(
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "variable_domain_size" => true,
                "values_onehot" => true,
            ), 
            [2, 5, board_size]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "values_onehot" => true,
            ), 
            [1, 6, board_size]
        ],
        [
            Dict(
                "constraint_activity" => true,
                "constraint_type" => true,
                "variable_initial_domain_size" => true,
                "variable_domain_size" => true,
                "values_raw" => true,
            ), 
            [2, 6, 1]
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
            [3, 7, 1]
        ],
    ]


    experiment_chosen_features_heterogeneous(board_size, n_episodes, n_instances;
        n_eval=n_eval,
        generator=nqueens_generator,
        chosen_features_list=chosen_features_list,
        type="nqueens",
        output_size=board_size,
        reward=reward)
end

###############################################################################
######### Experiment Type 5
#########  
######### 
###############################################################################
"""
Compares different action explorers with the heterogeneous representation for the nqueens problem.
"""
function experiment_explorer_heterogeneous_nqueens(board_size, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)

    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    experiment_explorer_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 6, 1], 
        output_size = board_size, 
        generator = nqueens_generator,  
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "nqueens",
        decay_steps=2000,
        c=2.0,
        basicHeuristics=nothing
    )
end

###############################################################################
######### Experiment Type 6
#########  
######### 
###############################################################################
"""
Compares different CPNNs with the heterogeneous representation for the nqueens problem.

"""
function experiment_nn_heterogeneous_nqueens(board_size, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)

    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    experiment_nn_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 6, 1], 
        output_size = board_size, 
        generator = nqueens_generator,
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "nqueens",
        decay_steps=2000,
        c=2.0,
        basicHeuristics=nothing
    )
end


###############################################################################
######### Experiment Type 8
#########  
######### 
###############################################################################


function experiment_chosen_features_hetcpnn_nqueens(chosen_features_list, board_size, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.NQueensGenerator(board_size)
    restartPerInstances = 1

    experiment_chosen_features_hetcpnn(
        board_size,
        board_size-5,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = board_size,
        update_horizon = board_size-5,
        generator = generator,
        chosen_features_list = chosen_features_list, 
        type = "nqueens_"*string(board_size)
        )
end
function experiment_chosen_features_hetffcpnn_nqueens(chosen_features_list, board_size, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.NQueensGenerator(board_size)
    restartPerInstances = 1

    experiment_chosen_features_hetffcpnn(
        board_size,
        board_size-5,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = board_size,
        update_horizon = board_size-5,
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type = "nqueens_"*string(board_size)
        )
end

###############################################################################
######### Experiment Type 9
#########  
######### 
###############################################################################

function experiment_transfer_heterogeneous_nqueens(board_size, 
    board_size_transfered, 
    n_episodes, 
    n_episodes_transfered, 
    n_instances; 
    n_layers_graph=3, 
    n_eval=10, 
    n_eval_transfered=10, 
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000)
    """
    
    """
    nqueens_generator = SeaPearl.NQueensGenerator(board_size)
    nqueens_generator_transfered = SeaPearl.NQueensGenerator(board_size_transfered)

    experiment_transfer_heterogeneous(board_size, board_size_transfered, n_episodes, n_episodes_transfered, n_instances;
        chosen_features=nothing,
        feature_size = [2, 6, 1], 
        output_size = board_size,
        output_size_transfered = board_size_transfered,
        generator = nqueens_generator, 
        generator_transfered = nqueens_generator_transfered,
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval,
        n_eval_transfered = n_eval_transfered, 
        reward = reward, 
        type = "nqueens",
        decay_steps=decay_steps,
        trajectory_capacity=trajectory_capacity,
    )
end

###############################################################################
######### Experiment Type 10
#########  
######### Restart
###############################################################################
function experiment_restart_heterogeneous_nqueens(board_size, n_episodes, n_instances;
    restart_list = [1, 5, 10, 20],
    n_layers_graph=3, 
    n_eval=10, 
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000)

    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    experiment_restart_heterogeneous(board_size, n_episodes, n_instances;
        restart_list = restart_list,
        feature_size = [2, 6, 1], 
        output_size = board_size,
        generator = nqueens_generator, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "nqueens",
        decay_steps=decay_steps,
        trajectory_capacity=trajectory_capacity
    )
end

###############################################################################
######### Experiment Type11
#########  
######### 
###############################################################################
"""
Compares different CPNNs with the heterogeneous representation for the nqueens problem.

"""
function experiment_activation_heterogeneous_nqueens(board_size, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)

    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    experiment_activation_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 6, 1], 
        output_size = board_size, 
        generator = nqueens_generator,
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "nqueens",
        decay_steps=2000,
        c=2.0,
        basicHeuristics=nothing
    )
end