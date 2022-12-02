@testset "classic_kidney_exchange.jl" begin
    SeaPearlZoo.solve_kidney_exchange_matrix(SeaPearlZoo.getKidneyExchangeInputData("classic_cp/kidney_exchange/kep_8_0.2"))
end