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

###############################################################################
######### Experiment Type MALIK
#########  
######### 
###############################################################################
"""
Compares different RL Agents with the heterogeneous representation for the nqueens problem.

"""
function experiment_rl_heterogeneous_nqueens(board_size, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)

    nqueens_generator = SeaPearl.NQueensGenerator(board_size)

    chosen_features = Dict(
        "variable_initial_domain_size" => true,
        "constraint_type" => true,
        "variable_domain_size" => true,
        "values_raw" => true)

    feature_size = [2,5,1]
    n_step_per_episode = Int(round(board_size*0.75))
    experiment_rl_heterogeneous(board_size, n_episodes, n_instances;
        chosen_features=chosen_features,
        feature_size = feature_size, 
        output_size = board_size, 
        generator = nqueens_generator,
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "nqueens",
        decay_steps=250*n_step_per_episode,
        basicHeuristics=nothing
    )
end

###############################################################################
######### Simple nqueens experiment
#########  
######### 
###############################################################################

function simple_experiment_nqueens(n, n_episodes, n_instances, variable_selection, chosen_features, feature_size; n_eval=10, eval_timeout=60)
    """
    Runs a single experiment on nqueens
    """
    n_step_per_episode = Int(round(n*0.7))
    reward = SeaPearl.GeneralReward
    generator = SeaPearl.NQueensGenerator(n)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    trajectory_capacity = 800*n_step_per_episode
    update_horizon = Int(round(n_step_per_episode//2))
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    agent_hetcpnn = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n),        
            get_explorer = () -> get_epsilon_greedy_explorer(250*n_step_per_episode, 0.0),
            batch_size=16,
            update_horizon=update_horizon,
            min_replay_history=Int(round(16*n_step_per_episode//2)),
            update_freq=1,
            target_update_freq=7*n_step_per_episode,
            get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=1,
                n_layers_graph=3,
                n_layers_node=2,
                n_layers_output=2
            )
        )
    learned_heuristic_hetffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_hetcpnn; chosen_features=chosen_features)
    learnedHeuristics["hetffcpnn"] = learned_heuristic_hetffcpnn
    variableHeuristic = nothing
    if variable_selection == "min"
        variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()
    elseif variable_selection == "random"
        variableHeuristic = SeaPearl.RandomVariableSelection{false}()
    else
        error("Variable selection method not implemented!")
    end
    
    basicHeuristics = OrderedDict(
        "random" => SeaPearl.RandomHeuristic()
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=1,
        eval_strategy = SeaPearl.ILDSearch(2),
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=0,
        exp_name= "nqueens_"*string(n)*"_heterogeneous_ffcpnn_"*string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing

end