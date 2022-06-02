using SeaPearl

###############################################################################
######### Experiment Type 1
#########  
######### 
###############################################################################

function experiment_representation(size, n_episodes, n_instances; feature_sizes, output_size, generator, expParameters, basicHeuristics=nothing, n_layers_graph=3, n_eval=10, reward=SeaPearl.GeneralReward, type="", chosen_features=nothing)
    """
    Compare three agents:
        - an agent with the default representation and default features;
        - an agent with the default representation and chosen features;
        - an agent with the heterogeneous representation and chosen features.
    """
    SR_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    agent_default_default = get_default_agent(;
        get_default_trajectory = () -> get_default_slart_trajectory(capacity=2000, n_actions=output_size),
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
        chosen_features = Dict(
            "constraint_activity" => true,
            "constraint_type" => true,
            "variable_initial_domain_size" => true,
            "variable_domain_size" => true,
            "values_raw" => true,
        )
    end

    agent_default_chosen = get_default_agent(;
        get_default_trajectory = () -> get_default_slart_trajectory(capacity=2000, n_actions=output_size),
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
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
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
        expParameters=expParameters,
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

function experiment_n_conv(n_nodes, n_min_color, density, n_episodes, n_instances; n_eval=10, generator, SR, chosen_features, feature_size, type="")
    """
    Compares the impact of the number of convolution layers for the heterogeneous representation.
    """
    get_agent = (SR <: SeaPearl.DefaultStateRepresentation) ? get_default_agent : get_heterogeneous_agent

    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:3
        agent = get_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
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
        expParameters=expParameters,
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

function experiment_chosen_features_heterogeneous(size, n_episodes, n_instances; output_size, n_layers_graph=3, n_eval=10, generator, chosen_features_list, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing, reward=SeaPearl.GeneralReward)
    """
    Compares the impact of the chosen_features for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:length(chosen_features_list)
        chosen_features = chosen_features_list[i][1]
        feature_size = chosen_features_list[i][2]
        agent = get_heterogeneous_agent(;
            get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),        
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
        expParameters=expParameters,
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

function experiment_heuristic_heterogeneous(size, n_episodes, n_instances; feature_size, output_size, n_eval=10, generator, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing, chosen_features=nothing, basicHeuristics, reward=SeaPearl.GeneralReward, n_layers_graph=3, eta_init=1.0, eta_stable=0.1, eta_decay_steps, helpValueHeuristic)
    """
    Compares the impact of simple vs supervised learned heuristic for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = Dict(
            "constraint_activity" => true,
            "constraint_type" => true,
            "variable_initial_domain_size" => true,
            "variable_domain_size" => true,
            "values_raw" => true,
        )
    end

    agent_simple = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
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
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
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
        expParameters=expParameters,
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

function experiment_explorer_heterogeneous(size, n_episodes, n_instances; feature_size, output_size, n_eval=10, generator, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing, chosen_features=nothing, basicHeuristics, reward=SeaPearl.GeneralReward, n_layers_graph=3, decay_steps=2000, c=2.0)
    """
    Compares the impact of the action explorer for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = Dict(
            "constraint_activity" => true,
            "constraint_type" => true,
            "variable_initial_domain_size" => true,
            "variable_domain_size" => true,
            "values_raw" => true,
        )
    end

    agent_epsilon_greedy = get_heterogeneous_agent(;
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
            n_layers_output=2
        )
    )
    learned_heuristic_epsilon_greedy = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_epsilon_greedy; chosen_features=chosen_features)

    agent_ucb = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
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
        expParameters=expParameters,
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

function experiment_nn_heterogeneous(size, n_episodes, n_instances; feature_size, output_size, n_eval=10, generator, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing, chosen_features=nothing, basicHeuristics, reward=SeaPearl.GeneralReward, n_layers_graph=3, decay_steps=2000, c=2.0)
    """
    Compares the impact of the action explorer for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = Dict(
            "constraint_activity" => true,
            "constraint_type" => true,
            "variable_initial_domain_size" => true,
            "variable_domain_size" => true,
            "values_raw" => true,
        )
    end

    agent_cpnn = get_heterogeneous_agent(;
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
            n_layers_output=2
        )
    )
    learned_heuristic_cpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_cpnn; chosen_features=chosen_features)

    agent_fullfeaturedcpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=2000, n_actions=output_size),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
        batch_size=16,
        update_horizon=8,
        min_replay_history=256,
        update_freq=1,
        target_update_freq=8,
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
    learned_heuristic_fullfeaturedcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_fullfeaturedcpnn; chosen_features=chosen_features)
    
    agent_variableoutputcpnn = get_heterogeneous_agent(;
        get_heterogeneous_trajectory = () -> get_heterogeneous_sart_trajectory(capacity=2000),
        get_explorer = () -> get_epsilon_greedy_explorer(decay_steps, 0.01),
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
    learned_heuristic_variableoutputcpnn = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.VariableOutput}(agent_variableoutputcpnn; chosen_features=chosen_features)
    

    learnedHeuristics = OrderedDict(
        "cpnn" => learned_heuristic_cpnn,
        "fullfeaturedcpnn" => learned_heuristic_fullfeaturedcpnn,
        "variableoutputcpnn" => learned_heuristic_variableoutputcpnn,
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
        expParameters=expParameters,
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

function experiment_pooling_heterogeneous(size, n_episodes, n_instances; feature_size, output_size, n_eval=10, generator, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing, chosen_features=nothing, basicHeuristics, reward=SeaPearl.GeneralReward, n_layers_graph=3, decay_steps=2000, c=2.0)
    """
    Compares the impact of the action explorer for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    if isnothing(chosen_features)
        chosen_features = Dict(
            "constraint_activity" => true,
            "constraint_type" => true,
            "variable_initial_domain_size" => true,
            "variable_domain_size" => true,
            "values_raw" => true,
        )
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
            output_size=1,
            n_layers_graph=n_layers_graph,
            n_layers_node=2,
            n_layers_output=2;
            pool=SeaPearl.meanPooling()
        )
    )
    learned_heuristic_mean = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent_mean; chosen_features=chosen_features)

    learnedHeuristics = OrderedDict(
        "sum" => learned_heuristic_sum,
        # "mean" => learned_heuristic_mean,
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
        expParameters=expParameters,
        nbRandomHeuristics=0,
        exp_name= type * "_heterogeneous_pooling_" * string(n_episodes) * "_" * string(size) * "_",
        eval_timeout=eval_timeout
    )
    nothing
end