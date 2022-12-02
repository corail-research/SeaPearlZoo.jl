@testset "classic_cp.jl" begin
    include("graph_coloring/graph_coloring.jl")
    include("eternity2/eternity2.jl")
    include("jobshop/jobshop.jl")
    include("kidney_exchange/kidney_exchange.jl")
    include("knapsack/knapsack.jl")
    include("latin/latin.jl")
    include("nqueens/nqueens.jl")
    include("tsptw/tsptw.jl")
end