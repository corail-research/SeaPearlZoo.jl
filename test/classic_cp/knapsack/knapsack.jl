@testset "knapsack.jl" begin
    Main.SeaPearlZoo.build_model_and_solve_knapsack(Main.SeaPearlZoo.parseKnapsackFile!("classic_cp/knapsack/ks_4_0"))
end