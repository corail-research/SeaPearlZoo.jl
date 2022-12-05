@testset "learning_nqueens.jl" begin
    board_size = 15
    features_type = SeaPearlZoo.BetterFeaturization
    SR = SeaPearl.DefaultStateRepresentation{features_type, SeaPearl.DefaultTrajectoryState}
    experiment_config = SeaPearlZoo.NQueensConfig(
        board_size, 
        SeaPearl.feature_length(SR), 
        1,
        100,
        50,
        0,
        1,
        SeaPearl.CPReward,
        false
    )
    model_config = SeaPearlZoo.NQueensModelConfig(SeaPearl.feature_length(SR), experiment_config.board_size, false)
    approximator_model = SeaPearlZoo.build_nqueens_model(model_config)
    target_approximator_model = SeaPearlZoo.build_nqueens_model(model_config)
    agent_config = SeaPearlZoo.NQueensAgentConfig(
        approximator_model,
        target_approximator_model,
        32,
        12,
        256,
        1,
        200,
        1.0,
        0.1,
        :exp,
        50000,
        1,
        50000,
        board_size
    )
    agent = SeaPearlZoo.build_nqueens_agent(agent_config)

    learned_heuristic_config = SeaPearlZoo.NQueensLearnedHeuristicConfig(1., 0.1 , 50, 50)
    learned_heuristic = SeaPearl.SupervisedLearnedHeuristic{SR, experiment_config.reward_type, SeaPearl.FixedOutput}(
        agent, 
        eta_init=learned_heuristic_config.eta_init,
        eta_stable=learned_heuristic_config.eta_stable, 
        warmup_steps=learned_heuristic_config.warmup_steps, 
        decay_steps=learned_heuristic_config.decay_steps,
        rng=MersenneTwister(1234)
    )

    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    random_heuristics = []
    for i in 1 : experiment_config.num_random_heuristics
        push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
    end

    value_selection_array = [learned_heuristic, heuristic_min]
    append!(value_selection_array, random_heuristics)
    variable_selection = SeaPearl.MinDomainVariableSelection{false}()

    SeaPearlZoo.solve_learning_nqueens(experiment_config, agent, learned_heuristic, variable_selection, value_selection_array)
end