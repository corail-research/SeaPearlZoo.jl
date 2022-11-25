using Dates
using Flux
using LightGraphs
import Pkg
using Random
using ReinforcementLearning
const RL = ReinforcementLearning


@testset "learning_cp.jl" begin
    include("eternity2/learning_eternity2.jl")
    include("graphcoloring/graphcoloring.jl")
    include("kep/kep.jl")
    include("knapsack/learning_knapsack.jl")
    include("latin/latin.jl")
end