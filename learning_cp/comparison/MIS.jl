include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 8
#########  
######### 
###############################################################################


function experiment_chosen_features_hetcpnn_MIS(chosen_features_list, n, k, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    restartPerInstances = 1

    experiment_chosen_features_hetcpnn(
        n,
        n,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = 2, 
        generator = generator,
        chosen_features_list = chosen_features_list, 
        type = "MIS_"*string(n)*"_"*string(k)
        )
end
function experiment_chosen_features_hetffcpnn_MIS(chosen_features_list, n, k, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    restartPerInstances = 1

    experiment_chosen_features_hetffcpnn(
        n,
        n,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = 2, 
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type = "MIS_"*string(n)*"_"*string(k)
        )
end

###############################################################################
######### Experiment Type 9
#########  
######### Transfer Learning
###############################################################################
function experiment_transfer_heterogeneous_mis(n, n_transfered, k, k_transfered, 
    n_episodes, 
    n_episodes_transfered, 
    n_instances; 
    n_layers_graph=3, 
    n_eval=10,
    n_eval_transfered=10,
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000, 
    eval_strategy=eval_strategy)
    mis_generator = SeaPearl.MaximumIndependentSetGenerator(n, k)
    mis_generator_transfered = SeaPearl.MaximumIndependentSetGenerator(n_transfered, k_transfered)
    
    # Basic value-selection heuristic
    basicHeuristics = OrderedDict(
        "maximum" => SeaPearl.BasicHeuristic() 
    )

    experiment_transfer_heterogeneous(n, n_transfered, n_episodes, n_episodes_transfered, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = 2,
        output_size_transfered = 2,
        generator = mis_generator, 
        generator_transfered = mis_generator_transfered,
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval,
        n_eval_transfered = n_eval_transfered,
        reward = reward, 
        type = "mis",
        decay_steps=decay_steps,
        trajectory_capacity=trajectory_capacity,
        eval_strategy=eval_strategy
    )
end


###############################################################################
######### Experiment Type 10
#########  
######### Restart
###############################################################################
function experiment_restart_heterogeneous_mis(n, k, n_episodes, n_instances;
    restart_list = [1, 5, 10, 20],
    n_layers_graph=3, 
    n_eval=10, 
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000)

    mis_generator = SeaPearl.MaximumIndependentSetGenerator(n, k)

    experiment_restart_heterogeneous(n, n_episodes, n_instances;
        restart_list = restart_list,
        feature_size = [2, 3, 1], 
        output_size = 2,
        generator = mis_generator, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval,
        reward = reward, 
        type = "MIS_"*string(n)*"_"*string(k),
        decay_steps=decay_steps,
        trajectory_capacity=trajectory_capacity
    )
end

###############################################################################
######### Experiment Type 11
#########  
######### 
###############################################################################
"""
Compares HGT and HeterogeneousGraphConv.
"""

function experiment_hgt_vs_graphconv_MIS(chosen_features, n, k, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    restartPerInstances = 1

    experiment_hgt_vs_graphconv(
        n,
        n,
        n_episodes,
        n_instances,
        restartPerInstances;
        output_size = 2, 
        generator = generator,
        chosen_features = chosen_features, 
        type = "MIS_"*string(n)*"_"*string(k)
        )
end