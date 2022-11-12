struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization,TS}) where TS
    g = sr.cplayergraph
    features = zeros(Float32, instance_generator.n+6, nv(g))
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

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization, TS}}) where TS
    instance_generator.n+6
end

function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization, TS}}) where TS
    return 0
end
