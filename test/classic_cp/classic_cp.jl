@testset "classic_cp.jl" begin
    include("coloring/coloring.jl")
    include("eternity2/eternity2.jl")
    include("jobshop/jobshop.jl")
    include("kidneyexchange/kidneyexchange.jl")
    include("knapsack/knapsack.jl")
    include("latin/latin.jl")
    include("nqueens/nqueens.jl")
    include("tsptw/tsptw.jl")
end