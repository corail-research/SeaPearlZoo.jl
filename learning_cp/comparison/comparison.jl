using SeaPearl

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
            capacity=2000,
            decay_steps=2000,
            ϵ_stable=0.01,
            batch_size=16,
            update_horizon=8,
            min_replay_history=256,
            update_freq=1,
            target_update_freq=8,
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=n_nodes,
            n_layers_graph=i,
            n_layers_node=2,
            n_layers_output=2
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
end

###############################################################################
######### Experiment Type 3
#########  
######### 
###############################################################################

function experiment_chosen_features_heterogeneous(size, n_episodes, n_instances; output_size, n_eval=10, generator, chosen_features_list, type="", expParameters=Dict{String,Any}()::Dict{String,Any}, eval_timeout=nothing)
    """
    Compares the impact of the chosen_features for the heterogeneous representation.
    """
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

    learnedHeuristics = OrderedDict{String,SeaPearl.LearnedHeuristic}()
    for i in 1:length(chosen_features_list)
        chosen_features = chosen_features_list[i][1]
        feature_size = chosen_features_list[i][2]
        agent = get_heterogeneous_agent(;
            capacity=2000,
            decay_steps=2000,
            ϵ_stable=0.01,
            batch_size=16,
            update_horizon=8,
            min_replay_history=256,
            update_freq=1,
            target_update_freq=8,
            feature_size=feature_size,
            conv_size=8,
            dense_size=16,
            output_size=output_size,
            n_layers_graph=i,
            n_layers_node=2,
            n_layers_output=2
        )
        learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)
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
end