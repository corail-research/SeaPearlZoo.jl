@testset "learning_nqueens.jl" begin
    experiment_config = SeaPearlZoo.TSPTWExperimentConfig(5, 10, 0, 100, 1, 200, 10, 1, 1, false)
    tsptw_generator = SeaPearl.TsptwGenerator(
        experiment_config.num_cities,
        experiment_config.grid_size,
        experiment_config.max_tw_gap,
        experiment_config.max_tw,
        true
    )

    num_features = 20
    featurizationType = SeaPearl.DefaultFeaturization

    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{featurizationType, TS}}) where TS
        return num_features
    end
    SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
    numInFeatures=SeaPearl.feature_length(SR)

    SR = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.DefaultTrajectoryState}
    num_input_features = SeaPearl.feature_length(SR)
    reward_type = SeaPearl.GeneralReward
    agent = SeaPearlZoo.build_tsptw_agent(num_input_features, experiment_config.num_cities)
    values_raw = true
    constraint_type = true
    chosen_features = Dict([("values_raw", values_raw), ("constraint_type", constraint_type)])
    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR, reward_type, SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)

    featurizationType = SeaPearl.DefaultFeaturization
    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{featurizationType, TS}}) where TS
        return num_input_features
    end

    random_heuristics = []
    for i in 1: experiment_config.num_random_heuristics
        push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
    end
    heuristic_min = SeaPearl.BasicHeuristic(SeaPearlZoo.select_min)
    value_selection_array = [learned_heuristic, heuristic_min]
    append!(value_selection_array, random_heuristics)
    variable_selection = SeaPearl.MinDomainVariableSelection{false}()

    metrics_array, eval_metrics_array = SeaPearlZoo.solve_tsptw_with_learning(experiment_config, value_selection_array, agent, learned_heuristic, variable_selection)
end