include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

###############################################################################
######### Experiment Type 4
#########  
######### Supervised vs Simple
###############################################################################

function experiment_heuristic_heterogeneous_mis(n, k, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)

    # Basic value-selection heuristic
    basicHeuristics = OrderedDict(
        "maximum" => SeaPearl.BasicHeuristic()
    )

    experiment_heuristic_heterogeneous(n, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = 2, 
        generator = generator,  
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "mis",
        eta_decay_steps = Int(floor(n_episodes/1.5)),
        helpValueHeuristic = SeaPearl.BasicHeuristic(),
        eta_init = 1.0,
        eta_stable = 0.0
    )
end

###############################################################################
######### Experiment Type 5
#########  
######### 
###############################################################################

function experiment_explorer_heterogeneous_MIS(chosen_features, feature_size, n, k, n_episodes, n_instances; n_eval=10)
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    restartPerInstances = 1

    selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    basicHeuristics = OrderedDict(
        "max" => heuristic_max
    )

    experiment_explorer_heterogeneous(
        n, 
        n,
        n_episodes, 
        n_instances; 
        feature_size = feature_size,
        chosen_features = chosen_features, 
        output_size = 2, 
        n_eval=n_eval, 
        generator, 
        type = "MIS_"*string(n)*"_"*string(k)*"_explorer_comparison", 
        basicHeuristics = basicHeuristics, 
        reward=SeaPearl.GeneralReward, 
        n_layers_graph=3, 
        c=2.0)
end

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

###############################################################################
######### Simple MIS experiment
#########  
######### 
###############################################################################

function simple_experiment_MIS(n, k, n_episodes, n_instances, chosen_features, feature_size; n_eval=10, eval_timeout=60)
    """
    Runs a single experiment on MIS
    """
    n_step_per_episode = Int(round(n//2))+k
    reward = SeaPearl.GeneralReward
    generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    trajectory_capacity = 800*n_step_per_episode
    update_horizon = Int(round(n_step_per_episode//2))
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    agent_hetcpnn = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=2),        
            get_explorer = () -> get_epsilon_greedy_explorer(500*n_step_per_episode, 0.05),
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
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()
    selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    basicHeuristics = OrderedDict(
        "max" => heuristic_max
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
        exp_name= "MIS_"*string(n)*"_"*string(k)*"_heterogeneous_ffcpnn_" * string(n_episodes),
        eval_timeout=eval_timeout
    )
    nothing

end

###############################################################################
######### Comparison of tripartite graph vs specialized graph
#########  
######### 
###############################################################################
"""
Compares the tripartite graph representation with a specific representation.
"""

function experiment_tripartite_vs_specific_MIS(n, k, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    
    MIS_generator = SeaPearl.MaximumIndependentSetGenerator(n,k)
    SR_specific = SeaPearl.MISStateRepresentation{SeaPearl.MISFeaturization,SeaPearl.DefaultTrajectoryState}
    
    # Basic value-selection heuristic
    selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
    heuristic_max = SeaPearl.BasicHeuristic(selectMax)
    basicHeuristics = OrderedDict(
        "max" => heuristic_max
    )

    chosen_features = Dict(
        "node_number_of_neighbors" => true,
        "constraint_type" => true,
        "constraint_activity" => true,
        "nb_not_bounded_variable" => true,
        "variable_initial_domain_size" => true,
        "variable_domain_size" => true,
        "variable_is_objective" => true,
        "variable_assigned_value" => true,
        "variable_is_bound" => true,
        "values_raw" => true)

    experiment_tripartite_vs_specific(n, n_episodes, n_instances, SR_specific;
    chosen_features = chosen_features,
    feature_size = [6, 5, 2],
    feature_size_specific = SeaPearl.feature_length(SR_specific),
    output_size = 2,
    generator = MIS_generator, 
    n_layers_graph = n_layers_graph,
    eval_strategy = SeaPearl.ILDSearch(2),
    n_eval = n_eval, 
    reward = reward, 
    type = "MIS",
    basicHeuristics=basicHeuristics
)
end

###############################################################################
######### Experiment Type MALIK
#########  
######### 
###############################################################################

"""
Compares different RL Agents with the heterogeneous representation for the MIS problem.
"""
function experiment_rl_heterogeneous_mis(n,k, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)

    mis_generator = SeaPearl.MaximumIndependentSetGenerator(n, k)

    chosen_features = Dict(
        "variable_initial_domain_size" => true,
        "constraint_type" => true,
        "variable_domain_size" => true,
        "values_raw" => true)

    feature_size = [2,2,1]
    n_step_per_episode = Int(round(n//2))+k
    experiment_rl_heterogeneous(n, n_episodes, n_instances;
        eval_strategy =  SeaPearl.ILDSearch(2),
        chosen_features=chosen_features,
        feature_size = feature_size, 
        output_size = 2, 
        generator = mis_generator,
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "mis",
        decay_steps=250*n_step_per_episode,
        basicHeuristics=nothing
    )
end