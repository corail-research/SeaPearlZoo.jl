using Dates
using Flux
using LightGraphs
import Pkg
using Random
using ReinforcementLearning
const RL = ReinforcementLearning


@testset "learning_cp.jl" begin
    include("eternity2/learning_eternity2.jl")
    include("graph_coloring/graph_coloring.jl")
    include("kidney_exchange/kidney_exchange.jl")
    include("knapsack/learning_knapsack.jl")
    include("latin/latin.jl")
    include("nqueens/nqueens.jl")
    include("tsptw/tsptw.jl")
end