@testset "learning_knapsack.jl" begin
    knapsack_generator = SeaPearl.KnapsackGenerator(10, 10, 0.2)
    StateRepresentation = SeaPearl.DefaultStateRepresentation{SeaPearlZoo.KnapsackFeaturization, SeaPearl.DefaultTrajectoryState}
    numInFeatures = SeaPearl.feature_length(StateRepresentation)
    experiment_setup = SeaPearlZoo.KnapsackExperimentConfig(1, 2, 1, 1)
    approximator_config = SeaPearlZoo.KnapsackApproximatorConfig(16, 1, SeaPearl.GraphConv(16 => 16, Flux.leakyrelu), false)
    target_approximator_config = SeaPearlZoo.KnapsackApproximatorConfig(16, 1, SeaPearl.GraphConv(16 => 16, Flux.leakyrelu), false)
    approximator_model = SeaPearlZoo.build_knapsack_approximator_model(approximator_config)
    target_approximator_model = SeaPearlZoo.build_knapsack_target_approximator_model(target_approximator_config)
    knapsack_agent_config = SeaPearlZoo.KnapsackAgentConfig( 0.9f0, 8, 10, 8, 8, 1, experiment_setup.num_episodes)
    agent = SeaPearlZoo.build_knapsack_agent(approximator_model, target_approximator_model, knapsack_agent_config)
    
    # Value Heuristic definition
    learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{StateRepresentation, SeaPearlZoo.knapsackReward, SeaPearl.FixedOutput}(agent)
    basicHeuristic = SeaPearl.BasicHeuristic((x; cpmodel=nothing) -> SeaPearl.maximum(x.domain)) 
    
    # Variable Heuristic definition
    struct KnapsackVariableSelection <: SeaPearl.AbstractVariableSelection{false} end
    
    function (::KnapsackVariableSelection)(model::SeaPearl.CPModel)
        i = 1
        while SeaPearl.isbound(model.variables["x[" * string(i) * "]"])
            i += 1
        end
        return model.variables["x[" * string(i) * "]"]
    end
    
    valueSelectionArray = [learnedHeuristic, basicHeuristic]
    
    function solve_knapsack_with_learning!(experiment_setup::SeaPearlZoo.KnapsackExperimentConfig, save_experiment_artefacts::Bool=false)
        experiment_parameters = Dict(
            :nbEpisodes => experiment_setup.num_episodes,
            :evalFreq => experiment_setup.eval_freq,
            :nbInstances => experiment_setup.num_instances
        )
        metricsArray, eval_metricsArray = SeaPearl.train!(;
            valueSelectionArray= valueSelectionArray,
            generator=knapsack_generator,
            nbEpisodes=experiment_setup.num_episodes,
            strategy=SeaPearl.DFSearch(),
            variableHeuristic=KnapsackVariableSelection(),
            out_solver=false,
            verbose=false,
            evaluator=SeaPearl.SameInstancesEvaluator(
                valueSelectionArray, 
                knapsack_generator; 
                evalFreq=experiment_setup.eval_freq, 
                nbInstances=experiment_setup.num_instances
            ),
            restartPerInstances = 1
        )

        return metricsArray, eval_metricsArray
    end
    
    metricsArray, eval_metricsArray = SeaPearlZoo.solve_knapsack_with_learning!(experiment_setup)
end