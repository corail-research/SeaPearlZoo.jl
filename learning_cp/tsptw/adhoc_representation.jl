mutable struct TsptwRepresentation <: SeaPearl.AbstractStateRepresentation
    dist::Matrix
    time_windows::Matrix
    current_city::Union{Nothing, Int64}
    possible_value_ids::Union{Nothing, Array{Int64}}
end

function TsptwRepresentation(model::SeaPearl.CPModel) where F
    dist = nothing
    tw_up = nothing
    tw_low = nothing
    for constraint in model.constraints
        if isnothing(dist) && isa(constraint, SeaPearl.Element2D) && size(constraint.matrix, 2) > 1
            dist = constraint.matrix
        end
        if isnothing(tw_low) && isa(constraint, SeaPearl.Element2D) && constraint.z.id == "lower_ai_1"
            tw_low = constraint.matrix
        end
        if isnothing(tw_up) && isa(constraint, SeaPearl.Element2D) && constraint.z.id == "upper_ai_1"
            tw_up = constraint.matrix
        end
    end

    sr = TsptwRepresentation(dist, hcat(tw_low, tw_up), nothing, nothing)
    sr
end

function SeaPearl.update_representation!(sr::TsptwRepresentation, model::SeaPearl.CPModel, x::SeaPearl.AbstractIntVar)
    sr.possible_value_ids = collect(x.domain)

    i = 1
    while x.id != "a_"*string(i)
        i += 1
    end
    if SeaPearl.isbound(model.variables["v_"*string(i)])
        sr.current_city = SeaPearl.assignedValue(model.variables["v_"*string(i)])
    end
    sr
end

function SeaPearl.to_arraybuffer(sr::TsptwRepresentation, rows=nothing::Union{Nothing, Int})::Array{Float32, 2}
    dist = sr.dist

    vector_values = zeros(Float32, size(dist, 1))
    for i in sr.possible_value_ids
        vector_values[i] = 1.
    end
    vector_current = zeros(Float32, size(dist, 1))
    if !isnothing(sr.current_city)
        vector_current[sr.current_city] = 1.
    end
    
    return hcat(sr.dist, sr.time_windows, vector_values, vector_current)
end
