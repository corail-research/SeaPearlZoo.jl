using SeaPearl

###############################################################################
######### Global Parameters
#########  
######### 
###############################################################################

DEFAULT_CHOSEN_FEATURES = Dict(
    "constraint_activity" => true,
    "constraint_type" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "values_raw" => true,
)

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
function experiment_representation(
    size, 
    n_episodes, 
    n_instances; 
    feature_sizes, 
    output_size, 
    generator, 
    basicHeuristics=nothing, 
    n_layers_graph=3, 
    n_eval=10, 
    reward=SeaPearl.GeneralReward, 
    type="", 
    chosen_features=nothing, 
    trajectory_capacity=2000
)
    
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    agent_default_default = get_default_agent(;
        get_default_trajectory = () -> get_default_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_default_nn = () -> get_default_cpnn(
            feature_size=feature_sizes[1],
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_default_default = SeaPearl.SimpleLearnedHeuristic{SR_default,reward,SeaPearl.FixedOutput}(agent_default_default)

    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_default_chosen = get_default_agent(;
        get_default_trajectory = () -> get_default_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_default_nn = () -> get_default_cpnn(
            feature_size=feature_sizes[2],
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_default_chosen = SeaPearl.SimpleLearnedHeuristic{SR_default,reward,SeaPearl.FixedOutput}(agent_default_chosen; chosen_features=chosen_features)

    agent_heterogeneous = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_sizes[3],
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_heterogeneous = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_heterogeneous; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "defaultdefault" => learned_heuristic_default_default,
        "defaultchosen" => learned_heuristic_default_chosen,
        "heterogeneous" => learned_heuristic_heterogeneous,
    )
    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name= type * "_representation_" * string(n_episodes) * "_" * string(size) * "_"
    )
    nothing
end

###############################################################################
######### Experiment Type 2
#########  
######### 
###############################################################################
"""
Compares the impact of the number of convolution layers for the heterogeneous representation.
"""

function experiment_n_conv(
    n_nodes, 
    n_episodes, 
    n_instances; 
    n_eval=10, 
    generator, 
    SR, 
    chosen_features, 
    feature_size, 
    type = "", 
    trajectory_capacity = 2000,
    output_size = n_nodes, 
    reward = SeaPearl.GeneralReward, 
    )

    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:3
        if SR <: SeaPearl.DefaultStateRepresentation
            agent = get_default_agent(;
            get_default_trajectory = () -> get_default_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
            get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
            batch_size=16,
            update_horizon=8,
            min_replay_history=256,
            update_freq=1,
            target_update_freq=8,
            get_default_nn = () -> get_default_cpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=i,
                n_layers_node=2,
                n_layers_output=2
            )
        )
        else 
            agent = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
            get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
            batch_size=16,
            update_horizon=8,
            min_replay_history=256,
            update_freq=1,
            target_update_freq=8,
            get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=i,
                n_layers_node=2,
                n_layers_output=2
            )
        )
        end
        
        if !isnothing(chosen_features)
            learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)
        else
            learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent)
        end
        learnedHeuristics[type * "_" *string(i)] = learned_heuristic
    end

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)

    basicHeuristics = OrderedDict(
        "min" => heuristic_min
    )

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name="graphcoloring_n_conv_" * type * "_" * string(n_episodes) * "_" * string(n_nodes) * "_"
    )
    nothing
end

###############################################################################
######### Experiment Type 3
#########  
######### 
###############################################################################
"""
Compares the impact of the chosen_features for the heterogeneous representation.
"""
function experiment_chosen_features_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    output_size = size, 
    n_layers_graph=3, 
    n_eval=10, 
    generator, 
    chosen_features_list, 
    type="",
    eval_timeout=nothing, 
    reward=SeaPearl.GeneralReward, 
    trajectory_capacity=2000
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:length(chosen_features_list)
        chosen_features = chosen_features_list[i][1]
        feature_size = chosen_features_list[i][2]
        agent = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),        
            get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
            batch_size=16,
            update_horizon=8,
            min_replay_history=256,
            update_freq=1,
            target_update_freq=8,
            get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=n_layers_graph,
                n_layers_node=2,
                n_layers_output=2
            )
        )
        learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)
        learnedHeuristics["heterogeneous_" *string(i)] = learned_heuristic
    end

    # Basic value-selection heuristic
    # selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    # heuristic_min = SeaPearl.BasicHeuristic(selectMin)

    basicHeuristics = OrderedDict(
        "min" => SeaPearl.RandomHeuristic()
    )

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_chosen_features_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 4
#########  
######### 
###############################################################################
"""
Compares the simple and the supervised learned heuristic for the heterogeneous representation.
"""
function experiment_heuristic_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    feature_size, 
    output_size, 
    n_eval=10, 
    generator, 
    type="", 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics, 
    reward=SeaPearl.GeneralReward, 
    trajectory_capacity=2000, 
    n_layers_graph=3, 
    eta_init=1.0, 
    eta_stable=0.1, 
    eta_decay_steps, 
    helpValueHeuristic
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_simple = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_simple = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_simple; chosen_features=chosen_features)

    agent_supervised = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=200,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_supervised = SeaPearl.SupervisedLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_supervised; chosen_features=chosen_features, eta_init=eta_init, eta_stable=eta_stable, decay_steps=eta_decay_steps, helpValueHeuristic=helpValueHeuristic)
    
    learnedHeuristics = OrderedDict(
        "simple" => learned_heuristic_simple,
        "supervised" => learned_heuristic_supervised,
    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_heuristic_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 5
#########  
######### 
###############################################################################
"""
Compares different action explorers for the heterogeneous representation.
"""
function experiment_explorer_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    feature_size, 
    output_size, 
    n_eval=10, 
    generator, 
    type="", 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics, 
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    decay_steps=2000, 
    c=2.0, 
    trajectory_capacity=2000
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_epsilon_greedy = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_epsilon_greedy = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_epsilon_greedy; chosen_features=chosen_features)

    agent_ucb = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_ucb_explorer(c, output_size),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_ucb = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ucb; chosen_features=chosen_features)
    
    learnedHeuristics = OrderedDict(
        "epsilon_greedy" => learned_heuristic_epsilon_greedy,
        "ucb" => learned_heuristic_ucb,
    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        exp_name= type * "_heterogeneous_explorer_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 6
#########  
######### 
###############################################################################
"""
Compares different CPNNs for the heterogeneous representation.
"""
function experiment_nn_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    feature_size, 
    output_size, 
    n_eval=10, 
    generator, 
    type="", 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics, 
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    decay_steps=n_episodes*size*0.8, 
    c=2.0, 
    trajectory_capacity=5000,
    pool=SeaPearl.sumPooling()
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
   
    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_cpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2,
            pool=pool
        )
    )
    learned_heuristic_cpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous, reward, SeaPearl.FixedOutput}(agent_cpnn; chosen_features=chosen_features)

    agent_fullfeaturedcpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2,
            pool=pool
        )
    )
    learned_heuristic_fullfeaturedcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous, reward, SeaPearl.FixedOutput}(agent_fullfeaturedcpnn; chosen_features=chosen_features)
    
    agent_ffcpnnv2 = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv2(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2
        )
    )
    learned_heuristic_ffcpnnv2 = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous, reward, SeaPearl.FixedOutput}(agent_ffcpnnv2; chosen_features=chosen_features)

    agent_ffcpnnv3 = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv3(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2
        )
    )
    learned_heuristic_ffcpnnv3 = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnnv3; chosen_features=chosen_features)
    
    agent_variableoutputcpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=size),
        get_explorer = () -> get_epsilon_greedy_explorer(1, 0),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_variableoutputcpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2
        )
    )
    learned_heuristic_variableoutputcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_variableoutputcpnn; chosen_features=chosen_features)
    

    learnedHeuristics = OrderedDict(
        #"cpnn" => learned_heuristic_cpnn,
        "fullfeaturedcpnn" => learned_heuristic_fullfeaturedcpnn,
        # "variableoutputcpnn" => learned_heuristic_variableoutputcpnn,
        # "ffcpnnv2" => learned_heuristic_ffcpnnv2,
        "ffcpnnv3" => learned_heuristic_ffcpnnv3
    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_cpnn_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 7
#########  
######### 
###############################################################################
"""
Compares different pooling methods in the CPNN for the heterogeneous representation.
"""
function experiment_pooling_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    feature_size, 
    output_size, 
    n_eval=10, 
    generator, 
    type="", 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics, 
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    decay_steps=2000, 
    c=2.0, 
    trajectory_capacity=2000
)
 
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_sum = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2;
            pool=SeaPearl.sumPooling()
        )
    )
    learned_heuristic_sum = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_sum; chosen_features=chosen_features)

    agent_mean = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
        get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2;
            pool=SeaPearl.meanPooling()
        )
    )
    learned_heuristic_mean = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_mean; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "sum" => learned_heuristic_sum,
        "mean" => learned_heuristic_mean,
    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        exp_name= type * "_heterogeneous_pooling_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 8
#########  
######### 
###############################################################################

"""
Compares different choices of features on HeterogeneousCPNN versus default_default
"""

function experiment_chosen_features_hetcpnn(
    size,
    n_step_per_episode,
    n_episodes,
    n_instances,
    restartPerInstances;
    output_size = size, 
    update_horizon = Int(round(n_step_per_episode//2)),
    n_layers_graph=3, 
    n_eval=10, 
    generator,
    chosen_features_list, 
    type="",
    eval_timeout=60, 
    reward=SeaPearl.GeneralReward, 
    trajectory_capacity=nothing,
    basicHeuristics = nothing
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
    trajectory_capacity = 500*n_step_per_episode
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:length(chosen_features_list)
        chosen_features = chosen_features_list[i][1]
        feature_size = chosen_features_list[i][2]
        agent_hetcpnn = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),        
            get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
            batch_size=32,
            update_horizon=update_horizon,
            min_replay_history=Int(round(32*n_step_per_episode//2)),
            update_freq=1,
            target_update_freq=7*n_step_per_episode,
            get_heterogeneous_nn = () -> get_heterogeneous_cpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=n_layers_graph,
                n_layers_node=2,
                n_layers_output=2
            )
        )
        learned_heuristic_hetcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_hetcpnn; chosen_features=chosen_features)
        
        learnedHeuristics["hetcpnn_" *string(i)] = learned_heuristic_hetcpnn
    end
    agent_default = get_default_agent(;
            get_default_trajectory = () -> get_default_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),        
            get_explorer = () -> get_epsilon_greedy_explorer(n_step_per_episode*300, 0.01),
            batch_size=32,
            update_horizon=Int(round(n_step_per_episode//2)),
            min_replay_history=Int(round(32*n_step_per_episode//2)),
            update_freq=1,
            target_update_freq=7*n_step_per_episode,
            get_default_nn = () -> get_default_cpnn(
                feature_size=3,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=n_layers_graph,
                n_layers_node=2,
                n_layers_output=2
            )
        )
    learned_heuristic_default = SeaPearl.SimpleLearnedHeuristic{SR_default,reward,SeaPearl.FixedOutput}(agent_default)
    learnedHeuristics["default"] = learned_heuristic_default

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_cpnn_chosen_features_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

function experiment_chosen_features_hetffcpnn(
    size,
    n_step_per_episode,
    n_episodes,
    n_instances,
    restartPerInstances;
    output_size = size, 
    n_layers_graph=3,
    n_eval=10, 
    generator,
    chosen_features_list,
    update_horizon = Int(round(n_step_per_episode//2)),
    type="",
    eval_timeout=60, 
    reward=SeaPearl.GeneralReward, 
    trajectory_capacity=nothing,
    basicHeuristics = nothing
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
    trajectory_capacity = 500*n_step_per_episode
    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:length(chosen_features_list)
        chosen_features = chosen_features_list[i][1]
        feature_size = chosen_features_list[i][2]
        agent_hetcpnn = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),        
            get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
            batch_size=32,
            update_horizon=update_horizon,
            min_replay_history=Int(round(32*n_step_per_episode//2)),
            update_freq=1,
            target_update_freq=7*n_step_per_episode,
            get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
                feature_size=feature_size,
                conv_size=8,
                dense_size=16,
                output_size=1,
                n_layers_graph=n_layers_graph,
                n_layers_node=2,
                n_layers_output=2
            )
        )
        learned_heuristic_hetffcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_hetcpnn; chosen_features=chosen_features)
        learnedHeuristics["hetffcpnn_" *string(i)] = learned_heuristic_hetffcpnn
    end
    agent_default = get_default_agent(;
            get_default_trajectory = () -> get_default_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),        
            get_explorer = () -> get_epsilon_greedy_explorer(n_step_per_episode*300, 0.01),
            batch_size=32,
            update_horizon=Int(round(n_step_per_episode//2)),
            min_replay_history=Int(round(32*n_step_per_episode//2)),
            update_freq=1,
            target_update_freq=7*n_step_per_episode,
            get_default_nn = () -> get_default_cpnn(
                feature_size=3,
                conv_size=8,
                dense_size=16,
                output_size=output_size,
                n_layers_graph=n_layers_graph,
                n_layers_node=2,
                n_layers_output=2
            )
        )
    learned_heuristic_default = SeaPearl.SimpleLearnedHeuristic{SR_default,reward,SeaPearl.FixedOutput}(agent_default)
    learnedHeuristics["default"] = learned_heuristic_default

    # -------------------
    # Variable Heuristic definition
    # -------------------
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        restartPerInstances=restartPerInstances,
        generator=generator,
        variableHeuristic=variableHeuristic,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        out_solver=true,
        verbose=true,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_ffcpnn_chosen_features" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 9
#########  
######### Transfer Learning
###############################################################################
function experiment_transfer_heterogeneous(
    size,
    size_transfered,
    n_episodes,
    n_episodes_transfered, 
    n_instances; 
    feature_size, 
    output_size,
    output_size_transfered, 
    n_eval=10, 
    generator,
    generator_transfered, 
    type="", 
    expParameters=Dict{String,Any}()::Dict{String,Any}, 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics=nothing, 
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    decay_steps=2000, 
    trajectory_capacity=2000,
    update_horizon=8,
    min_replay_history=128,
    verbose=true,
)
 
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
            feature_size=feature_size,
            n_layers_graph=n_layers_graph
        )
    )
    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "ffcppn" => learned_heuristic,
    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes,
        evalFreq=Int(floor(n_episodes / n_eval)),
        nbInstances=n_instances,
        generator=generator,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        verbose=verbose,
        exp_name= type * "_transfer_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )

    agent_transfer = RL.Agent(
        policy= RL.QBasedPolicy(
            learner=deepcopy(agent.policy.learner),
            explorer= get_epsilon_greedy_explorer(decay_steps, 0.01),
        ),
        trajectory=get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size_transfered)
    )
    learned_heuristic_transfer = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_transfer; chosen_features=chosen_features)

    agent = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size_transfered),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
            feature_size=feature_size,
            n_layers_graph=n_layers_graph
        )
    )
    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "ffcppn" => learned_heuristic,
        "transfer" => learned_heuristic_transfer,
    )

    metricsArray, eval_metricsArray = trytrain(
        nbEpisodes=n_episodes_transfered,
        evalFreq=Int(floor(n_episodes_transfered / n_eval)),
        nbInstances=n_instances,
        generator=generator_transfered,
        learnedHeuristics=learnedHeuristics,
        basicHeuristics=basicHeuristics;
        verbose=verbose,
        exp_name= type * "_transfered_" * string(n_episodes_transfered) * "_" * string(size_transfered) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end

###############################################################################
######### Experiment Type 10
#########  
######### Restart
###############################################################################
function experiment_restart_heterogeneous(
    size, 
    n_episodes, 
    n_instances;
    restart_list = [1, 5, 10, 20],
    output_size = size, 
    n_eval=10, 
    generator, 
    type="",
    eval_timeout=nothing, 
    verbose = false,
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    trajectory_capacity=2000,
    decay_steps = 2000,
    update_horizon = 8,
    min_replay_history = 128,
    feature_size,
    chosen_features=nothing,
    basicHeuristics=nothing
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    
    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end
    
    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    
    for i in 1:length(restart_list)
        n_restart = restart_list[i]
        
        learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
        agent = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
            get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
                feature_size=feature_size,
                n_layers_graph=n_layers_graph
            )
        )
        learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)
        learnedHeuristics["heterogeneous_" *string(i)] = learned_heuristic
    
        trytrain(
            nbEpisodes=n_episodes,
            evalFreq=Int(floor(n_episodes / n_eval)),
            nbInstances=n_instances,
            restartPerInstances=n_restart,
            generator=generator,
            learnedHeuristics=learnedHeuristics,
            basicHeuristics=basicHeuristics;
            verbose=verbose,
            exp_name= type * "_restart_" * string(i) * "_" * string(n_episodes) * "_" * string(size) * "_",
            eval_timeout=eval_timeout
        )
    end
    nothing
end


###############################################################################
######### Experiment Type 11
#########  
######### 
###############################################################################
"""
Compares different activation functions on the dense network for the heterogeneous representation.
"""
function experiment_activation_heterogeneous(
    size, 
    n_episodes, 
    n_instances; 
    feature_size, 
    output_size, 
    n_eval=10, 
    generator, 
    type="", 
    eval_timeout=nothing, 
    chosen_features=nothing, 
    basicHeuristics, 
    reward=SeaPearl.GeneralReward, 
    n_layers_graph=3, 
    decay_steps=n_episodes*size*0.8, 
    c=2.0, 
    trajectory_capacity=5000,
    pool=SeaPearl.sumPooling()
)

    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
   
    if isnothing(chosen_features)
        chosen_features = DEFAULT_CHOSEN_FEATURES
    end

    agent_fullfeaturedcpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2,
            pool=pool
        )
    )
    learned_heuristic_fullfeaturedcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous, reward, SeaPearl.FixedOutput}(agent_fullfeaturedcpnn; chosen_features=chosen_features)
    
    agent_ffcpnnv3_relu = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv3(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2,
            σ=NNlib.relu
        )
    )
    learned_heuristic_ffcpnnv3_relu = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnnv3_relu; chosen_features=chosen_features)
    
    agent_ffcpnnv3_sigmoid = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv3(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2,
            σ=NNlib.sigmoid
        )
    )

    learned_heuristic_ffcpnnv3_sigmoid = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnnv3_sigmoid; chosen_features=chosen_features)

    agent_ffcpnnv3_leakyrelu = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv3(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2,
            σ=NNlib.leakyrelu
        )
    )
    learned_heuristic_ffcpnnv3_leakyrelu = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnnv3_leakyrelu; chosen_features=chosen_features)
    
    agent_ffcpnnv3_id = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=trajectory_capacity, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=10,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=80,
        get_heterogeneous_nn = () -> get_heterogeneous_ffcpnnv3(
            feature_size=feature_size,
            conv_size=16,
            dense_size=16,
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_output=2,
            σ=identity
        )
    )
    learned_heuristic_ffcpnnv3_id = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_ffcpnnv3_id; chosen_features=chosen_features)
    
    learnedHeuristics = OrderedDict(
        #"cpnn" => learned_heuristic_cpnn,
        #"fullfeaturedcpnn" => learned_heuristic_fullfeaturedcpnn,
        # "variableoutputcpnn" => learned_heuristic_variableoutputcpnn,
        # "ffcpnnv2" => learned_heuristic_ffcpnnv2,
        "ffcpnnv3_relu" => learned_heuristic_ffcpnnv3_relu,
        "ffcpnnv3_sigmoid" => learned_heuristic_ffcpnnv3_sigmoid,
        "ffcpnnv3_leakyrelu" => learned_heuristic_ffcpnnv3_leakyrelu,
        "ffcpnnv3_id" => learned_heuristic_ffcpnnv3_id,


    )

    if isnothing(basicHeuristics)
        basicHeuristics = OrderedDict(
            "random" => SeaPearl.RandomHeuristic()
        )
    end
    variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

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
        verbose=false,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_cpnn_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end