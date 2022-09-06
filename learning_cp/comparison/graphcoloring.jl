include("../common/experiment.jl")
include("../common/utils.jl")
include("comparison.jl")

using CUDA
###############################################################################
######### simple GC experiment
#########  
######### 
###############################################################################
###############################################################################
######### simple GC experiment
#########  
######### 
###############################################################################
function simple_graph_coloring_experiment(n_nodes, n_nodes_eval, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=20, reward = SeaPearl.GeneralReward, c=2.0, trajectory_capacity=10000, pool = SeaPearl.sumPooling(), nbRandomHeuristics = 1, eval_timeout = nothing, restartPerInstances = 10, seedEval = nothing)

    n_step_per_episode = n_nodes
    update_horizon = Int(round(n_step_per_episode//2))

    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    eval_coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes_eval, n_min_color, density)

    evalFreq=Int(floor(n_episodes / n_eval))


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

    feature_size = [6, 6, 2]

    decay_steps = Int(floor(n_episodes*restartPerInstances*(n_nodes+1)*0.5))

    rngExp = MersenneTwister(seedEval)
    init = Flux.glorot_uniform(MersenneTwister(seedEval))

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

        # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    rngExp = MersenneTwister(seedEval)
    init = Flux.glorot_uniform(MersenneTwister(seedEval))

    agent_ffcpnn = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n_nodes),        
    get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.05; rng = rngExp ),
    batch_size=32,
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
        n_layers_output=2, 
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init
    ),
    γ = 0.99f0
    )

    agent_cpnn = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n_nodes),        
    get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01; rng = rngExp ),
    batch_size=16,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=1,
    target_update_freq=8*n_nodes,
    get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
        feature_size=feature_size,
        conv_size=8,
        dense_size=8,
        output_size=n_nodes, 
        n_layers_graph=n_layers_graph,
        n_layers_output=2,
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init,
        #device =gpu
    ),
    γ = 0.99f0
    )

    learned_heuristic_ffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnn, chosen_features=chosen_features)
    #learned_heuristic_control = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_control; chosen_features=chosen_features)
    #learned_heuristic_cpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_cpnn; chosen_features=chosen_features)
     
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()

    learnedHeuristics["ffcpnn"] = learned_heuristic_ffcpnn
    #learnedHeuristics["control"] = learned_heuristic_control
    #learnedHeuristics["cpnn"] = learned_heuristic_cpnn
    
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=evalFreq,
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        generator=coloring_generator,
#       eval_strategy=SeaPearl.ILDSearch(2),
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name="graph_coloring_benchmark" * string(n_nodes) *"_"*string(n_nodes_eval) * "_" * string(n_episodes) * "_",
        eval_timeout=eval_timeout, 
        eval_generator=eval_coloring_generator,
        training_timeout = 1800,
        #eval_every = 5,
    )

end
###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################
"""
Compare three agents:
    - an agent with the default representation and default features;
    - an agent with the default representation and chosen features;
    - an agent with the heterogeneous representation and chosen features.
"""
function experiment_representation_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
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
"""
Compares the impact of the number of convolution layers for the heterogeneous representation.
"""
function experiment_heterogeneous_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
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

"""
Compares the impact of the number of convolution layers for the default representation with chosen features.
"""
function experiment_default_chosen_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
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

"""
Compares the impact of the number of convolution layers for the default representation.
"""
function experiment_default_default_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
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
"""
Compares the impact of the chosen features for the heterogeneous representation.
"""
function experiment_chosen_features_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
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
"""
Compares the simple and the supervised learned heuristic for the heterogeneous representation.
"""
function experiment_heuristic_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
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

"""
Compares different action explorers for the heterogeneous representation.
"""
function experiment_explorer_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
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

function experiment_nn_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances, n_step_per_episode; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward, pool = SeaPearl.sumPooling(), restartPerInstances = 1)
    """
    Compare agents with different Fullfeatured CPNN pipeline
    """
    
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
        :pooling => string(pool)
    )

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_nn_heterogeneous(n_nodes, n_step_per_episode, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        c=2.0,
        basicHeuristics=basicHeuristics,
        pool = pool, 
        restartPerInstances = restartPerInstances
    )
end

function experiment_nn_heterogeneous_graphcoloringv4(n_nodes, n_min_color, density, n_episodes, n_instances, n_step_per_episode; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward, pool = SeaPearl.sumPooling())
    """
    Compare agents with different Fullfeatured CPNN pipeline
    """
    
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
        :pooling => string(pool)
    )

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_nn_heterogeneousv4(n_nodes, n_step_per_episode, n_episodes, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes, 
        generator = coloring_generator, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        c=2.0,
        basicHeuristics=basicHeuristics,
        pool = pool
    )
end

###############################################################################
######### Experiment Type 7
#########  
######### 
###############################################################################

"""
Compares different pooling methods in the CPNN for the heterogeneous representation.
"""
function experiment_pooling_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
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

"""
Compares different choices of features on HeterogeneousCPNN versus default_default
"""
function experiment_chosen_features_hetcpnn_graphcoloring(chosen_features_list, n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    restartPerInstances = 1
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_chosen_features_hetcpnn(
        n_nodes,
        n_nodes+1,
        n_episodes,
        n_instances,
        restartPerInstances;
        basicHeuristics = basicHeuristics,
        output_size = n_nodes, 
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type="graphcoloring_"*string(n_nodes),
        )
end


"""
Compares different choices of features on HeterogeneousFullFeaturedCPNN versus default_default
"""
function experiment_chosen_features_hetffcpnn_graphcoloring(chosen_features_list, n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10)
    generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    restartPerInstances = 1
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_chosen_features_hetffcpnn(
        n_nodes,
        n_nodes+1,
        n_episodes,
        n_instances,
        restartPerInstances;
        basicHeuristics = basicHeuristics,
        output_size = n_nodes, 
        generator=generator,
        chosen_features_list=chosen_features_list, 
        type="graphcoloring_"*string(n_nodes)
        )
end

###############################################################################
######### Experiment Type 9
#########  
######### Transfer Learning
###############################################################################
"""
Tests the impact of transfer learning
"""
function experiment_transfer_heterogeneous_graphcoloring(n_nodes, 
    n_nodes_transfered, 
    n_min_color, 
    density, 
    n_episodes, 
    n_episodes_transfered, 
    n_instances; 
    n_layers_graph=3, 
    n_eval=10,
    n_eval_transfered=10, 
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000)
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    coloring_generator_transfered = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes_transfered, n_min_color, density)
    

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_transfer_heterogeneous(n_nodes, n_nodes_transfered, n_episodes, n_episodes_transfered, n_instances;
        chosen_features=nothing,
        feature_size = [2, 3, 1], 
        output_size = n_nodes,
        output_size_transfered = n_nodes_transfered,
        generator = coloring_generator, 
        generator_transfered = coloring_generator_transfered,
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        n_eval_transfered = n_eval_transfered,
        reward = reward, 
        type = "graphcoloring",
        decay_steps=decay_steps,
        trajectory_capacity=trajectory_capacity
    )
end

###############################################################################
######### Experiment Type 10
#########  
######### Restart
###############################################################################
"""
Compares different values of argument `restartPerInstances``
"""

function experiment_restart_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances;
    restart_list = [1, 5, 10, 20],
    n_layers_graph=3, 
    n_eval=10, 
    reward=SeaPearl.GeneralReward, 
    decay_steps=2000, 
    trajectory_capacity=2000)

    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_restart_heterogeneous(n_nodes, n_episodes, n_instances;
        restart_list = restart_list,
        feature_size = [2, 3, 1], 
        output_size = n_nodes,
        generator = coloring_generator, 
        basicHeuristics = basicHeuristics, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
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
Compare different activation functions.
"""
function experiment_activation_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances;
    n_layers_graph=3,
    n_eval=10,
    reward=SeaPearl.GeneralReward,
    pool = SeaPearl.sumPooling()
    )
    
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
        :pooling => string(pool)
    )

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_activation_heterogeneous(n_nodes, n_episodes, n_instances;
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
        basicHeuristics=basicHeuristics,
        pool = pool
    )
end

###############################################################################
######### Experiment Type 12
#########  
######### 
###############################################################################
"""
Compare different pooling functions for the graph features in the different versions of FFCPNN.
"""

function experiment_features_pooling_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward, pool = SeaPearl.sumPooling())
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
        :pooling => string(pool)
    )

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    experiment_features_pooling_heterogeneous(n_nodes, n_episodes, n_instances;
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
    basicHeuristics=basicHeuristics,
    pool = pool
)
end

###############################################################################
######### Simple graphcoloring experiment
#########  
######### 
###############################################################################
"""
Runs a single experiment on graphcoloring
"""

function simple_experiment_graphcoloring(n, k, n_episodes, n_instances, chosen_features, feature_size; n_eval=10, eval_timeout=60)
    n_step_per_episode = n
    reward = SeaPearl.GeneralReward
    generator = SeaPearl.BarabasiAlbertGraphGenerator(n,k)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    trajectory_capacity = 800*n_step_per_episode
    update_horizon = Int(round(n_step_per_episode//2))
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    agent_hetcpnn = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n),        
            get_explorer = () -> get_epsilon_greedy_explorer(250*n_step_per_episode, 0.01),
            batch_size=16,
            update_horizon=update_horizon,
            min_replay_history=Int(round(16*n_step_per_episode//2)),
            update_freq=n_step_per_episode,
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
    variableHeuristic = SeaPearl.MinDomainVariableSelection{true}()

    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=1,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=0,
        exp_name= "graphcoloring_"*string(n)*"_heterogeneous_ffcpnn_" * string(n_episodes),
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment update_freq comparison
#########  
######### 
###############################################################################
"""
Compares different values of  argument `update_freq`
"""
function experiment_update_freq_graphcoloring(n_nodes, n_step_per_episode, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward, pool = SeaPearl.sumPooling())
    
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    
    expParameters = Dict(
        :generatorParameters => Dict(
            :nbNodes => n_nodes,
            :nbMinColor => n_min_color,
            :density => density
        ),
        :pooling => string(pool)
    )
    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )
    experiment_update_freq(n_nodes, n_episodes, n_step_per_episode, n_instances;
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
        basicHeuristics=basicHeuristics,
        pool = pool
    )
end

###############################################################################
######### Comparison of tripartite graph vs specialized graph
#########  
######### 
###############################################################################
"""
Compares the tripartite graph representation with a specific representation.
"""

function experiment_tripartite_vs_specific_graphcoloring(n, k, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    
    coloring_generator = SeaPearl.BarabasiAlbertGraphGenerator(n, k)
    SR_specific = SeaPearl.GraphColoringStateRepresentation{SeaPearl.GraphColoringFeaturization,SeaPearl.DefaultTrajectoryState}
    
    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
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

    experiment_tripartite_vs_specific(n, n, n_episodes, n_instances, SR_specific;
    chosen_features = chosen_features,
    feature_size = [6, 5, 2],
    feature_size_specific = SeaPearl.feature_length(SR_specific),
    output_size = n,
    generator = coloring_generator, 
    n_layers_graph = n_layers_graph, 
    n_eval = n_eval, 
    reward = reward, 
    type = "graphcoloring",
    basicHeuristics=basicHeuristics
)
end

###############################################################################
######### Experiment Type MALIK
#########  
######### 
###############################################################################

function experiment_rl_heterogeneous_graphcoloring(n_nodes, n_min_color, density, n_episodes, n_instances; n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward)
    """
    Compare rl agents
    """
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    chosen_features = Dict(
        "variable_initial_domain_size" => true,
        "constraint_type" => true,
        "variable_domain_size" => true,
        "values_raw" => true)

    feature_size = [2,2,1]
    
    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )
    n_step_per_episode = Int(round(n_nodes*0.75))
    experiment_rl_heterogeneous(n_nodes, n_episodes, n_instances;
        chosen_features=chosen_features,
        feature_size = feature_size, 
        output_size = n_nodes, 
        generator = coloring_generator, 
        basicHeuristics = nothing, 
        n_layers_graph = n_layers_graph, 
        n_eval = n_eval, 
        reward = reward, 
        type = "graphcoloring",
        decay_steps=250*n_step_per_episode,
    )
end

###############################################################################
######### Reward comparison experiment
#########  
######### 
###############################################################################

function reward_comparison_graphcoloring(n, density, min_nodes, n_episodes, n_instances, chosen_features, feature_size; n_eval=10, eval_timeout=60)
    """
    Runs a single experiment on graphcoloring
    """
    n_step_per_episode = n
    reward1 = SeaPearl.GeneralReward
    reward2 = SeaPearl.CPReward 
    generator = SeaPearl.ClusterizedGraphColoringGenerator(n,min_nodes,density)
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    trajectory_capacity = 800*n_step_per_episode
    update_horizon = Int(round(n_step_per_episode//2))
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    agent_gen = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n),        
            get_explorer = () -> get_epsilon_greedy_explorer(250*n_step_per_episode, 0.01),
            batch_size=16,
            update_horizon=update_horizon,
            min_replay_history=Int(round(16*n_step_per_episode//2)),
            update_freq=n_step_per_episode,
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

        agent_cp = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n),        
        get_explorer = () -> get_epsilon_greedy_explorer(250*n_step_per_episode, 0.01),
        batch_size=16,
        update_horizon=update_horizon,
        min_replay_history=Int(round(16*n_step_per_episode//2)),
        update_freq=n_step_per_episode,
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
    learned_heuristic_gen = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward1,SeaPearl.FixedOutput}(agent_gen; chosen_features=chosen_features)
    learnedHeuristics["gen"] = learned_heuristic_gen
    learned_heuristic_cp = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward2,SeaPearl.FixedOutput}(agent_cp; chosen_features=chosen_features)
    learnedHeuristics["cp"] = learned_heuristic_cp
    variableHeuristic = SeaPearl.MinDomainVariableSelection{true}()

    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=1,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=0,
        exp_name= "graphcoloring_"*string(n)*"_heterogeneous_ffcpnn_" * string(n_episodes),
        eval_timeout=eval_timeout
    )
    nothing
end


function transfert_graph_coloring_experiment(n_nodes, n_nodes_eval, k, n_episodes, n_instances; n_layers_graph=3, n_eval=25, reward = SeaPearl.GeneralReward, c=2.0, trajectory_capacity = 30000, pool = SeaPearl.meanPooling(), nbRandomHeuristics = 1, eval_timeout = 240, restartPerInstances = 1, seedEval = nothing, device = gpu, batch_size = 64, update_freq = 10,  target_update_freq= 500, name = "", numDevice = 0, eval_strategy = SeaPearl.DFSearch())
    n_step_per_episode = n_nodes

    update_horizon = Int(round(n_step_per_episode//2))

    if device == gpu
        CUDA.device!(numDevice)
    end
    coloring_generator = SeaPearl.BarabasiAlbertGraphGenerator(n_nodes,k)
    eval_coloring_generator = SeaPearl.BarabasiAlbertGraphGenerator(n_nodes_eval,k)
    n_min_color = k
    density = 0.9
    coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)
    eval_coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes_eval, n_min_color, density)

    evalFreq=Int(floor(n_episodes / n_eval))

    step_explorer = Int(floor(n_episodes*n_step_per_episode*0.1 ))

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

    feature_size = [6, 5, 2]

    rngExp = MersenneTwister(seedEval)
    init = Flux.glorot_uniform(MersenneTwister(seedEval))

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

        # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    agent_3 = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=n_nodes),        
    get_explorer = () -> get_epsilon_greedy_explorer(step_explorer, 0.05; rng = rngExp ),
    batch_size=batch_size,
    update_horizon=update_horizon,
    min_replay_history=Int(round(16*n_step_per_episode//2)),
    update_freq=update_freq,
    target_update_freq=target_update_freq,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=feature_size,
        conv_size=8,
        dense_size=16,
        output_size=1,
        n_layers_graph=3,
        n_layers_node=3,
        n_layers_output=2, 
        pool=pool,
        σ=NNlib.leakyrelu,
        init = init,
        device = device
    ),
    γ =  0.99f0
    )

    learned_heuristic_3 = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_3; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "3layer" => learned_heuristic_3,
    )

    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=evalFreq,
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        generator=coloring_generator,
        eval_strategy=eval_strategy,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=false,
        seedEval=seedEval,
        nbRandomHeuristics=nbRandomHeuristics,
        exp_name=name *'_'* string(n_nodes) *"_"*string(n_nodes_eval) * "_" * string(n_episodes) * "_"* string(seedEval) * "_",
        eval_timeout=eval_timeout, 
        eval_generator=eval_coloring_generator,
        device = device
    )

end
