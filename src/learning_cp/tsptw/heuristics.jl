function select_nearest_neighbor(x::SeaPearl.IntVar; cpmodel::Union{SeaPearl.CPModel, Nothing}=nothing)
    i = 1
    while "a_"*string(i) != x.id
        i += 1
    end
    current_city = SeaPearl.assignedValue(cpmodel.variables["v_"*string(i)])
    dist = SeaPearl.find_tsptw_dist_matrix(cpmodel)

    j = 0
    minDist = 0
    closer = j
    found_one = false
    while j < size(dist, 1)
        j += 1
        if (!found_one || dist[current_city, j] < minDist) && j in x.domain
            minDist = dist[current_city, j]
            closer = j
            found_one = true
        end
    end
    return closer
end

struct TsptwVariableSelection{TakeObjective} <: SeaPearl.AbstractVariableSelection{TakeObjective} end
TsptwVariableSelection(;take_objective=false) = TsptwVariableSelection{take_objective}()
function (::TsptwVariableSelection{false})(cpmodel::SeaPearl.CPModel; rng=nothing)
    for i in 1:length(keys(cpmodel.variables))
        if haskey(cpmodel.variables, "a_"*string(i)) && !SeaPearl.isbound(cpmodel.variables["a_"*string(i)])
            return cpmodel.variables["a_"*string(i)]
        end
    end
end