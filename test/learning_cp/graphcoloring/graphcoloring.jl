@testset "learning_coloring.jl" begin
    coloring_settings = ColoringExperimentSettings(1, 1, 100, 1, 1, 20, 5, 0.95)
    instance_generator = SeaPearl.BarabasiAlbertGraphGenerator(coloring_settings.nbNodes, coloring_settings.nbMinColor)

    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, TS}}) where TS
        instance_generator.n + 6
    end

    function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, TS}}) where TS
        return 0
    end

    function SeaPearl.featurize(
        sr::SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization,TS}
        ) where TS

        g = sr.cplayergraph
        features = zeros(Float32, instance_generator.n + 6, nv(g))
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.VariableVertex)
                features[1,i] = 1.
                if g.cpmodel.objective == cp_vertex.variable
                    features[6, i] = 1.
                end
            end
            if isa(cp_vertex, SeaPearl.ConstraintVertex)
                features[2, i] = 1.
                constraint = cp_vertex.constraint
                if isa(constraint, SeaPearl.NotEqual)
                    features[4, i] = 1.
                end
                if isa(constraint, SeaPearl.LessOrEqual)
                    features[5, i] = 1.
                end
            end
            if isa(cp_vertex, SeaPearl.ValueVertex)
                features[3, i] = 1.
                value = cp_vertex.value
                features[6+value, i] = 1.
            end
        end
        features
    end

    rewardType = SeaPearl.CPReward
    SR = SeaPearl.DefaultStateRepresentation{SeaPearl.AbstractFeaturization, SeaPearl.DefaultTrajectoryState}
    numInFeatures = SeaPearl.feature_length(SR)

    # -------------------
    # Value Heuristic definition
    # -------------------
    output_size = instance_generator.n
    agent_config = ColoringAgentConfig(0.99f0, 16, output_size, 4, 128, 1, 200, 3000)
    approximator_model = build_graph_coloring_approximator_model(instance_generator.n)
    target_approximator_model = build_graph_coloring_target_approximator_model(instance_generator.n)
    agent = build_graph_coloring_agent(approximator_model, target_approximator_model, agent_config)
    learnedHeuristic = SeaPearl.SimpleLearnedHeuristic{SR, rewardType, SeaPearl.FixedOutput}(agent)

    # Basic value-selection heuristic
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
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

    randomHeuristics = []
    for i in 1:coloring_settings.nbRandomHeuristics
        push!(randomHeuristics, SeaPearl.BasicHeuristic(select_random_value))
    end

    valueSelectionArray = [learnedHeuristic, heuristic_min]
    append!(valueSelectionArray, randomHeuristics)
    variableSelection = SeaPearl.MinDomainVariableSelection{false}() # Variable Heuristic definition

    metricsArray, eval_metricsArray = solve_learning_coloring(agent, agent_config, coloring_settings, instance_generator)
    
end