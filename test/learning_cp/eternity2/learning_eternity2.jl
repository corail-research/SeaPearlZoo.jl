@testset "learning_eternity2.jl" begin
    eternity2_generator = SeaPearl.Eternity2Generator(6, 6, 6)
    struct EternityFeaturization <: SeaPearl.AbstractFeaturization end
    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
        return 6
    end
    function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
        return 0
    end
    SR = SeaPearl.DefaultStateRepresentation{EternityFeaturization, SeaPearl.DefaultTrajectoryState}
    num_input_features = SeaPearl.feature_length(SR)

    experiment_config = SeaPearlZoo.ExperimentConfig(eternity2_generator, 100, 30, 1, 1)
    model_config = SeaPearlZoo.EternityModelConfig(Flux.leakyrelu, false, num_input_features)
    approximator_model = SeaPearlZoo.build_approximator_model(model_config)
    target_approximator_model = SeaPearlZoo.build_approximator_model(model_config)
    eternity_agent_config = SeaPearlZoo.EternityAgentConfig(
        eternity2_generator.m, 
        eternity2_generator.n,
        approximator_model,
        target_approximator_model
    )
    dqn_learner_config = SeaPearlZoo.DQNLearnerConfig(0.9f0, 8, 7, 8, 8, 100)
    agent = SeaPearlZoo.build_eternity2_agent(dqn_learner_config, eternity_agent_config, experiment_config.num_episodes)

    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{SR, SeaPearl.CPReward, SeaPearl.FixedOutput}(agent)
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    min_domain_heuristic = SeaPearl.BasicHeuristic(selectMin)

    function select_random_value(x::SeaPearl.IntVar; cpmodel=nothing)
        selected_number = rand(1:length(x.domain))
        i = 1
        for value in x.domain
            if i == selected_number
                return value
            end
            i += 1
        end
        @assert false "This should not happen"
    end

    random_heuristics = []
    for i in 1: experiment_config.num_random_heuristics
        push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
    end

    value_selection_array = [learned_heuristic, min_domain_heuristic]
    append!(value_selection_array, random_heuristics)
    variable_selection = SeaPearl.MinDomainVariableSelection{false}()

    metrics_array, eval_metrics_array = SeaPearlZoo.train_eternity2_model(experiment_config, value_selection_array, variable_selection)
end