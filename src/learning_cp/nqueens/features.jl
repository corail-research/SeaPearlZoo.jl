struct BetterFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{BetterFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, 2*nqueens_generator.board_size+1, LightGraphs.nv(g))
    for i in 1:LightGraphs.nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            if isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
                id = cp_vertex.variable.id
                if occursin("+",id)
                    features[parse(Int, split(id,"+")[end]), i] = 1
                else
                    features[parse(Int, split(id,"-")[end]), i] = -1
                end
                
            else
                id = cp_vertex.variable.id
                features[string_to_queen(id), i] = 1
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[nqueens_generator.board_size+1, i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[3, i] = 1.
            value = cp_vertex.value
            features[nqueens_generator.board_size+1+value, i] = 1.
        end
    end
    features
end

function string_to_queen(id::String)::Int
    parse(Int, split(id,"_")[end])
end

#initializing phase
#update_with_cp_model

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{BetterFeaturization, TS}}) where TS
    return 1 + 2*board_size
end
