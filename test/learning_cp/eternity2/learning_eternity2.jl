@testset "learning_eternity2.jl" begin
    eternity2_generator = SeaPearl.Eternity2Generator(6, 6, 6)
    struct EternityFeaturization <: SeaPearl.AbstractFeaturization end
    function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}) where TS
        g = sr.cplayergraph
        pieces  = g.cpmodel.adhocInfo
        features = zeros(Float32, 6, nv(g))
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.VariableVertex)
                id = cp_vertex.variable.id
                prefix, suffix = split(id,'_')
                if prefix=="id"
                    a, b = parse(Int, suffix[1]), parse(Int, suffix[2])
                    features[1,i] = a
                    features[2,i] = b
                    features[3,i] = -1
                    features[4,i] = -1
                    features[5,i] = -1
                    features[6,i] = -1
                end
            elseif isa(cp_vertex, SeaPearl.ValueVertex)
                v = cp_vertex.value
                piece = pieces[v,:]
                features[1,i] = piece[1]
                features[2,i] = piece[2]
                features[3,i] = piece[3]
                features[4,i] = piece[4]
            end
        end
        features
    end
    
    function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}, model::SeaPearl.CPModel) where TS
        g = sr.cplayergraph
        features = sr.nodeFeatures
        for i in 1:nv(g)
            cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
            if isa(cp_vertex, SeaPearl.VariableVertex)
                id = cp_vertex.variable.id
                prefix, suffix = split(id,'_')
                if prefix=="id"
                    a, b = parse(Int, suffix[1]), parse(Int, suffix[2])
                    var = sr.cplayergraph.cpmodel.variables["src_v"*string(a)*string(b)]
                    features[3,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                    var = sr.cplayergraph.cpmodel.variables["src_v"*string(a)*string(b+1)]
                    features[4,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                    var = sr.cplayergraph.cpmodel.variables["src_h"*string(a)*string(b)]
                    features[5,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                    var = sr.cplayergraph.cpmodel.variables["src_h"*string(a+1)*string(b)]
                    features[6,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                end
            end
        end
    end
    function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
        return 6
    end
    function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
        return 0
    end
    SR = SeaPearl.DefaultStateRepresentation{EternityFeaturization, SeaPearl.DefaultTrajectoryState}
    num_input_features = SeaPearl.feature_length(SR)

    experiment_config = SeaPearlZoo.ExperimentConfig(eternity2_generator, 1, 3, 1, 1)
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