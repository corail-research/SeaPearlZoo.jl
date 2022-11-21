import Pkg
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Dates
using Random
using LightGraphs

@testset "learning_cp.jl" begin
    include("eternity2/learning_eternity2.jl")
    include("graphcoloring/graphcoloring.jl")
    include("knapsack/learning_knapsack.jl")
end