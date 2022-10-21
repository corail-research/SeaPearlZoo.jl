using SeaPearl
include("../src/SeaPearlZoo.jl")
using Test

@testset "SeaPearlZoo" begin
    include("classic_cp/classic_cp.jl")
end

revise_user = "You use Revise, you're efficient in your work, well done ;)"