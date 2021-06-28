struct EternityFeaturization <: SeaPearl.AbstractFeaturization end
struct EternityFeaturization2 <: SeaPearl.AbstractFeaturization end
struct EternityFeaturization3 <: SeaPearl.AbstractFeaturization end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    features = zeros(Float32, 17, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex) && !isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
            id = cp_vertex.variable.id
            #println(id)
            s = split(id,['_'])[1]
            l = s[1]
            t = split(id,['_'])[2]
            if l=='o'#orientation
                features[1,i] = 1
                a,b = parse(Int, split(t,"")[1]), parse(Int, split(t,"")[2])
                features[2,i] = a
                features[3,i] = b
            elseif l=='i'
                features[8,i] = 1
                features[9,i] = parse(Int, split(t,"")[1])
                features[10,i] = parse(Int, split(t,"")[2])
            else
                if t[1]=="v" features[11,i] = 1 else features[12,i] = 1 end
                features[13,i] = parse(Int, t[2])
                features[14,i] = parse(Int, t[3])
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[15,i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[16,i] = cp_vertex.value
        end
        features[17,i] = 0
    end
    #sr.cplayergraph.cpmodel.adhocInfo = zeros(Int, 1+nqueens_generator.board_size)
    features
end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization3,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    features = zeros(Float32, 9, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex) && !isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
            id = cp_vertex.variable.id
            #println(id)
            s = split(id,['_'])[1]
            l = s[1]
            t = split(id,['_'])[2]
            if l=='i'
                features[1,i] = 1
                features[2,i] = parse(Int, split(t,"")[1])
                features[3,i] = parse(Int, split(t,"")[2])
            else
                if t[1]=="v" features[4,i] = 1 else features[5,i] = 1 end
                features[6,i] = parse(Int, t[2])
                features[7,i] = parse(Int, t[3])
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[8,i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[9,i] = cp_vertex.value
        end
        #features[17,i] = 0
    end

    #sr.cplayergraph.cpmodel.adhocInfo = zeros(Int, 1+nqueens_generator.board_size)
    features
end

function SeaPearl.globalFeaturize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization3,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    feature = zeros(Float32, n*m)
    feature
end

function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization2,TS}) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    k = eternity2_generator.k
    g = sr.cplayergraph
    features = zeros(Float32, 12, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex) && !isa(cp_vertex.variable, SeaPearl.IntVarViewOffset)
            id = cp_vertex.variable.id
            #println(id)
            s = split(id,['_'])[1]
            l = s[1]
            t = split(id,['_'])[2]
            if l=='o'#orientation
                features[1,i] = 1
                a,b = parse(Int, split(t,"")[1]), parse(Int, split(t,"")[2])
                features[2,i] = a
                features[3,i] = b
            elseif l=='i'
                features[4,i] = 1
                features[5,i] = parse(Int, split(t,"")[1])
                features[6,i] = parse(Int, split(t,"")[2])
            else
                if t[1]=="v" features[7,i] = 1 else features[8,i] = 1 end
                features[9,i] = parse(Int, t[2])
                features[10,i] = parse(Int, t[3])
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[11,i] = 1
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[12,i] = cp_vertex.value
        end
        #features[17,i] = 0
    end
    #sr.cplayergraph.cpmodel.adhocInfo = zeros(Int, 1+nqueens_generator.board_size)
    features
end

function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization3,TS}, model::SeaPearl.CPModel) where TS
    n = eternity2_generator.n
    m = eternity2_generator.m
    g = sr.cplayergraph
    vars_branchable = SeaPearl.branchable_variables(model)
    feature = sr.globalFeature
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            id = cp_vertex.variable.id
            var = cp_vertex.variable
            if !isa(cp_vertex.variable, SeaPearl.IntVarViewOffset) && split(id,['_'])[1][1]=='o'
                t = split(id,['_'])[2]
                a,b = parse(Int, t[1]), parse(Int, t[2])
                feature[(a-1)*m + b] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : 0
            end
        end
    end
end

function SeaPearl.update_features!(sr::SeaPearl.DefaultStateRepresentation{EternityFeaturization,TS}, model::SeaPearl.CPModel) where TS
    g = sr.cplayergraph
    vars_branchable = SeaPearl.branchable_variables(model)
    features = sr.features
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            id = cp_vertex.variable.id
            if !isa(cp_vertex.variable, SeaPearl.IntVarViewOffset) && split(id,['_'])[1][1]=='i'
                t = split(id,['_'])[2]
                a,b = parse(Int, t[1]), parse(Int, t[2])
                var = sr.cplayergraph.cpmodel.variables["src_v"*string(a)*string(b)]
                features[7,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                var = sr.cplayergraph.cpmodel.variables["src_v"*string(a)*string(b+1)]
                features[5,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                var = sr.cplayergraph.cpmodel.variables["src_h"*string(a)*string(b)]
                features[4,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                var = sr.cplayergraph.cpmodel.variables["src_h"*string(a+1)*string(b)]
                features[6,i] = SeaPearl.isbound(var) ? SeaPearl.assignedValue(var) : -1
                features[17,i] = model.statistics.numberOfNodes
            end
        end
    end
end

#initializing phase
#update_with_cp_model
#

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization, TS}}) where TS
    return 17
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization2, TS}}) where TS
    return 12
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization3, TS}}) where TS
    return 9
end

function SeaPearl.globalFeature_length(::Type{SeaPearl.DefaultStateRepresentation{EternityFeaturization3, TS}}) where TS
    return eternity2_generator.n*eternity2_generator.m
end