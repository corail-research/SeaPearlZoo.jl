@testset "eternity2.jl" begin
    eternityInput::SeaPearlZoo.EternityInputData = SeaPearlZoo.parseEternityInput("classic_cp/eternity2/eternity3x3")
    model = SeaPearlZoo.model_eternity2(eternityInput)
    variableSelection = SeaPearl.MinDomainVariableSelection{false}()
    valueSelection = SeaPearl.BasicHeuristic()
    status = @time SeaPearl.solve!(model; variableHeuristic=variableSelection, valueSelection=valueSelection)
    # SeaPearlZoo.solve_eternity2("classic_cp/eternity2/eternity3x3")
end