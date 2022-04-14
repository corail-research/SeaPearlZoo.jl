using LightGraphs

struct KepFeaturization <: SeaPearl.AbstractFeaturization end

numberOfFeatures = 8
"""
    SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{KepFeaturization})

The default featurise function from the SeaPearl Package is overwritten to fit the kep problem topology
Here, a node feature encodes the following informations :
[
    1 :   the node represents a VARIABLE
    2 :    |__ the variable type is "OBJECTIVE FUNCTION"
    3 :    |__ the variable type is "IntVarViewOpposite"
    4 :   the node represents a CONSTRAINT
    5 :    |__ the constraint type is "SumToZero"
    6 :    |__ the constraint type is "SumLessThan"
    7 :   the node represents a VALUE
    8 :   |__ the variable value
]
"""
function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{KepFeaturization})
    g = sr.cplayergraph
    features = zeros(Float32, numberOfFeatures, nv(g))
    for i in 1:nv(g)
        cp_vertex = SeaPearl.cpVertexFromIndex(g, i)
        if isa(cp_vertex, SeaPearl.VariableVertex)
            variable = cp_vertex.variable
            features[1,i] = 1.
            if g.cpmodel.objective == variable
                features[2,i] = 1.
            end

            if isa(variable, SeaPearl.IntVarViewOpposite)
                features[3, i] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[4, i] = 1.
            constraint = cp_vertex.constraint
            if isa(constraint, SeaPearl.SumToZero)
                features[5, i] = 1.
            end
            if isa(constraint, SeaPearl.SumLessThan)
                features[6, i] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[7, i] = 1.
            value = cp_vertex.value
            features[8, i] = value 
        end
    end
    features
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{KepFeaturization, TS}}) where TS
    numberOfFeatures
end

