struct EternityFeaturization <: SeaPearl.AbstractFeaturization end

#this is the best featurization for eternity2 so far.
#it uses FullFeaturedCPNN

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    pieces  = g.cpmodel.adhocInfo
    features = zeros(Float32, 6, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            id = cp_vertex.variable.id
            #println(id)
            prefix, suffix = split(id,'_')
            if prefix=="id"
                a,b = parse(Int, suffix[1]), parse(Int, suffix[2])
                features[1,i] = a
                features[2,i] = b
                features[3,i] = -1
                features[4,i] = -1
                features[5,i] = -1
                features[6,i] = -1
                #features[7,i] = length(cp_vertex.variable.domain)
                #features[8,i] = length(cp_vertex.variable.onDomainChange)
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
                a,b = parse(Int, suffix[1]), parse(Int, suffix[2])
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
"""
function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
    return 11
end

function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
    return 0
end
"""
