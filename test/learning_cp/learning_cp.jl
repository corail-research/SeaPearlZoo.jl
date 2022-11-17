import Pkg
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using Dates
using Random
using LightGraphs

@testset "learning_cp.jl" begin
    include("graphcoloring/graphcoloring.jl")
    println("hahahah")
end