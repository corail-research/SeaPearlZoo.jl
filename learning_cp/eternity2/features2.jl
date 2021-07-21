struct EternityFeaturization <: SeaPearl.AbstractFeaturization end
"""
node features:
1 => node is id
2 => node is horizontal edge
3 => node is vertical edge
4 => vertical index
5 => horizontal index
6 => domain length
7 => # constraints linked
8 => assigned value
9 => node is constraint
10 => node value
11 => node degree

global features:
None
"""



function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    features = zeros(Float32, 11, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            id = cp_vertex.variable.id
            #println(id)
            prefix, suffix = split(id,'_')
            if prefix=="id"
                features[1,i] = 1
            elseif prefix=="srch"
                features[2,i] = 1
            elseif prefix=="srcv"
                features[3,i] = 1 
            end
            features[4,i] = parse(Int, suffix[1])
            features[5,i] = parse(Int, suffix[2])
            features[6,i] = length(cp_vertex.variable.domain)
            features[7,i] = length(cp_vertex.variable.onDomainChange)
        elseif isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[9,i] = 1
        elseif isa(cp_vertex, SeaPearl.ValueVertex)
            features[10,i] = cp_vertex.value
        end
        features[11,i] = degree(sr.cplayergraph, i)
    end
    features
end

function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}, model::SeaPearl.CPModel) where TS
    g = sr.cplayergraph
    features = sr.nodeFeatures
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            var = cp_vertex.variable
            features[6,i] = length(var.domain)
            if SeaPearl.isbound(var)
                features[7,i] = SeaPearl.assignedValue(var)
            end
        end
        features[11,i] = degree(g, i)
    end
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
    return 11
end

function SeaPearl.global_feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
    return 0
end
