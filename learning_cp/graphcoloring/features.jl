struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, nv(g), coloring_generator.n+6)
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            features[i, 1] = 1.
            if g.cpmodel.objective == cp_vertex.variable
                features[i, 6] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[i, 2] = 1.
            constraint = cp_vertex.constraint
            if isa(constraint, SeaPearl.NotEqual)
                features[i, 4] = 1.
            end
            if isa(constraint, SeaPearl.LessOrEqual)
                features[i, 5] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[i, 3] = 1.
            value = cp_vertex.value
            features[i, 6+value] = 1.
        end
    end
    features
end

function SeaPearl.feature_length(gen::SeaPearl.ClusterizedGraphColoringGenerator, ::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization}})
    return 6 + gen.n
end
