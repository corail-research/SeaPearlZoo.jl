@testset "latin.jl" begin
    tile::Matrix{Int} = zeros(Int64, (3, 3))
    model = SeaPearlZoo.model_latin(tile)
    SeaPearlZoo.solve_latin!(model)
end