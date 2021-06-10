struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, nv(g), nqueens_generator.board_size+3)
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            if isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
                features[i, 1] =0
            else
                id = cp_vertex.variable.id
                features[i, 1] = string_to_queen(id)/nqueens_generator.board_size
            end
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

function string_to_queen(id::String)::Int
    parse(Int, split(id,"_")[end])
end

#initializing phase
#update_with_cp_model
#

function SeaPearl.feature_length(gen::SeaPearl.NQueensGenerator, ::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization}})
    return 3 + gen.board_size
end
