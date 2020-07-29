function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization}) 
    g = sr.cplayergraph
    features = zeros(Float32, nv(g), numberOfFeatures) 
    for i in 1:nv(g) 
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i) 
        if isa(cp_vertex, SeaPearl.VariableVertex) 
            variable = cp_vertex.variable 
            features[i, 1] = 1. 
            if g.cpmodel.objective == variable
                features[i, 2] = 1. 
            end 
            
            if isa(variable, SeaPearl.IntVarViewOpposite)
                features[i, 3] = 1. 
            end
            if isa(variable, SeaPearl.IntVarViewMul)
                features[i, 4] = 1. 
                features[i, 5] = variable.a 
            end
        end 
        if isa(cp_vertex, SeaPearl.ConstraintVertex) 
            features[i, 6] = 1. 
            constraint = cp_vertex.constraint 
            if isa(constraint, SeaPearl.SumToZero) 
                features[i, 7] = 1. 
            end 
            if isa(constraint, SeaPearl.LessOrEqualConstant) 
                features[i, 8] = 1. 
            end 
        end 
        if isa(cp_vertex, SeaPearl.ValueVertex) 
            features[i, 9] = 1. 
            value = cp_vertex.value 
            features[i, 10] = value/10 #max_weight
        end 
    end 
    features 
    # features = zeros(Float32, nv(g), nv(g)) 
    # for i in 1:size(features)[1] 
    #     features[i, i] = 1.0f0 
    # end 
    # features 
end 