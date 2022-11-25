@testset "learning_latin.jl" begin
    state_representation= SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization, SeaPearl.DefaultTrajectoryState}
    latin_exp_config = SeaPearlZoo.LatinExperimentConfig(
        state_representation,
        1,
        3,
        1,
        1,
        0,
        11,
        0.6
    )

    function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization,TS}) where TS
        g = sr.cplayergraph
        features = zeros(Float32, 7, nv(g))
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.VariableVertex)
                features[1,i]=1
                id = cp_vertex.variable.id
                t = split(id,['_'])[2]
                a,b = parse(Int, split(t,",")[1]), parse(Int, split(t,",")[2]) #coordinates on the board
                features[2,i],features[3,i] = a,b
            elseif isa(cp_vertex, SeaPearl.ConstraintVertex)
                if isa(cp_vertex.constraint, SeaPearl.EqualConstant)
                    features[5,i] = cp_vertex.constraint.v
                else
                    features[6,i] = 1
                end
            elseif isa(cp_vertex, SeaPearl.ValueVertex)
                features[7,i] = cp_vertex.value
            end
        end
        features
    end

    function SeaPearl.global_featurize(sr::SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization,TS}) where TS
        N = latin_exp_config.generator.N
        g = sr.cplayergraph
        feature = zeros(Float32, N^2)
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.ConstraintVertex)&&isa(cp_vertex.constraint, SeaPearl.EqualConstant)
                x = cp_vertex.constraint.x
                id=x.id
                v = cp_vertex.constraint.v
                t = split(id,['_'])[2]
                a,b = parse(Int, split(t,",")[1]), parse(Int, split(t,",")[2])
                feature[(a-1)*N + b] = v
            end
        end
        feature
    end

    function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization,TS}, model::SeaPearl.CPModel) where TS
        N = latin_exp_config.generator.N
        g = sr.cplayergraph
        globalFeature = sr.globalFeatures
        features = sr.nodeFeatures
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.VariableVertex)
                var = cp_vertex.variable
                id = cp_vertex.variable.id
                t = split(id,['_'])[2]
                a,b = parse(Int, split(t,",")[1]), parse(Int, split(t,",")[2])
                if SeaPearl.isbound(var)
                    features[4, i] = SeaPearl.assignedValue(var)
                    globalFeature[(a-1)*N + b] = SeaPearl.assignedValue(var)
                else
                    features[4, i] = 0
                    globalFeature[(a-1)*N + b] = 0
                end
            end
        end
    end

    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization, TS}}) where TS
        return 7
    end

    function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{SeaPearlZoo.LatinFeaturization, TS}}) where TS
        return latin_exp_config.generator.N * latin_exp_config.generator.N
    end

    model_config = SeaPearlZoo.LatinModelConfig(
        SeaPearl.GraphConv(32 => 32, Flux.leakyrelu),
        2,
        SeaPearl.feature_length(state_representation),
        SeaPearl.global_feature_length(state_representation),
        latin_exp_config.generator.N,
        latin_exp_config.generator.p,
        false,
    )
    approximator_model = SeaPearlZoo.build_latin_model(model_config)
    target_approximator_model = SeaPearlZoo.build_latin_model(model_config)
    agent_config = SeaPearlZoo.LatinAgentConfig(
        latin_exp_config.generator.N,
        approximator_model,
        target_approximator_model,
        16,
        3,
        10,
        8,
        100,
        0.1,
        2000,
        1,
        1000
    )
    agent = SeaPearlZoo.build_latin_agent(agent_config)

    learned_heuristic = SeaPearl.SimpleLearnedHeuristic{state_representation, SeaPearlZoo.InspectLatinReward, SeaPearl.FixedOutput}(agent)
    selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
    heuristic_min = SeaPearl.BasicHeuristic(selectMin)
    random_heuristics = []

    for i in 1 : latin_exp_config.num_random_heuristics
        push!(random_heuristics, SeaPearl.BasicHeuristic(select_random_value))
    end

    value_selection_array = [learned_heuristic, heuristic_min]
    append!(value_selection_array, random_heuristics)
    variable_selection = SeaPearl.MinDomainVariableSelection{false}()
    SeaPearlZoo.train_latin_agent(agent, latin_exp_config, value_selection_array, variable_selection, false)
end