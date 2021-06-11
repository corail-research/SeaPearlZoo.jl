struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, nqueens_generator.board_size+3, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            if isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
                features[1, i] =0
            else
                id = cp_vertex.variable.id
                features[1, i] = string_to_queen(id)/nqueens_generator.board_size
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[2, i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[3, i] = 1.
            value = cp_vertex.value
            features[3+value, i] = 1.
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
