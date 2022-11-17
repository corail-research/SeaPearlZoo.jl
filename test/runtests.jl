using SeaPearl
using SeaPearlZoo
using Test

@testset "SeaPearlZoo" begin
   include("classic_cp/classic_cp.jl")
   include("learning_cp/learning_cp.jl")
end

revise_user = "You use Revise, you're efficient in your work, well done ;)" 