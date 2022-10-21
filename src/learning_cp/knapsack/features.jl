using LightGraphs

struct KnapsackFeaturization <: SeaPearl.AbstractFeaturization end

numberOfFeatures =10
"""
    SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})

The default featurise function from the SeaPearl Package is overwritten to fit the knapsack problem topology
Here, a node feature encodes the following informations :
[
    1 :   the node represents a VARIABLE
    2 :    |__ the variable type is "OBJECTIVE FUNCTION"
    3 :    |__ the variable type is "IntVarViewOpposite"
    4 :    |__ the variable type is "IntVarViewMul"
    5 :          |__ encodes the multiplicative factor a
    6 :   the node represents a CONSTRAINT
    7 :    |__ the constraint type is "SumToZero"
    8 :    |__ the constraint type is "LessOrEqualConstant"
    9 :   the node represents a VALUE
    10 :   |__ the variable value / 10
]
"""
function SeaPearl.featurize(sr::SeaPearl.DefaultStateRepresentation{KnapsackFeaturization})
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
            if isa(variable, SeaPearl.IntVarViewMul)
                features[4, i] = 1.
                features[5, i] = variable.a
            end
        end
        if isa(cp_vertex, SeaPearl.ConstraintVertex)
            features[6, i] = 1.
            constraint = cp_vertex.constraint
            if isa(constraint, SeaPearl.SumToZero)
                features[7, i] = 1.
            end
            if isa(constraint, SeaPearl.LessOrEqualConstant)
                features[8, i] = 1.
            end
        end
        if isa(cp_vertex, SeaPearl.ValueVertex)
            features[9, i] = 1.
            value = cp_vertex.value
            features[10, i] = value/10 #max_weight
        end
    end
    features
end

function SeaPearl.feature_length(::Type{SeaPearl.DefaultStateRepresentation{KnapsackFeaturization, TS}}) where TS
    numberOfFeatures
end
