struct LatinFeaturization <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{LatinFeaturization,TS}) where TS
    N = latin_generator.N
    p = latin_generator.p
    g = sr.cplayergraph
    features = zeros(Float32, 7, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            features[1,i]=1
            id = cp_vertex.variable.id
            t = split(id,['_'])[2]
            a,b = parse(Int, split(t,",")[1]), parse(Int, split(t,",")[2])
            features[2,i],feature[3,i] = a,b
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

function SeaPearl.globalFeaturize(sr::SeaPearl.DefaultStateRepresentation{LatinFeaturization,TS}) where TS
    N = latin_generator.N
    p = latin_generator.p
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
            feature[(a-1)*n + b] = v
        end
    end
    feature
end

function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{LatinFeaturization,TS}, model::SeaPearl.CPModel) where TS
    N = latin_generator.N
    p = latin_generator.p
    g = sr.cplayergraph
    vars_branchable = SeaPearl.branchable_variables(model)
    globalFeature = sr.globalFeature
    features = sr.features
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            var = cp_vertex.variable
            id = cp_vertex.variable.id
            t = split(id,['_'])[2]
            a,b = parse(Int, split(t,",")[1]), parse(Int, split(t,",")[2])
            if SeaPearl.isbound(var)
                features[i,4] = SeaPearl.assignedValue(var)
                globalFeature[(a-1)*n + b] = SeaPearl.assignedValue(var)
            else
                features[i,4] = 0
                globalFeature[(a-1)*n + b] = 0
            end
        end
    end
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{LatinFeaturization, TS}}) where TS
    return 7
end

function SeaPearl.globalFeature_length(::Type{SeaPearl.DefaultStateRepresentation{LatinFeaturization, TS}}) where TS
    return latin_generator.N*latin_generator.N
end