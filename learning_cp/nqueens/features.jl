struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, nv(g), nqueens_generator.board_size+3)
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            features[i, 1] = 1
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[i, 2] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[i, 3] = 1.
            value = cp_vertex.value
            features[i, 3+value] = 1.
        end
    end
    features
end

function SeaPearl.feature_length(gen::SeaPearl.NQueensGenerator, ::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization}})
    return 3 + gen.board_size
end
