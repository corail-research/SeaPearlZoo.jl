@testset "kidneyexchange.jl" begin
    Main.SeaPearlZoo.solve_kidneyexchange_matrix(Main.SeaPearlZoo.getKidneyExchangeInputData("classic_cp/kidneyexchange/kep_8_0.2"))
end