# This util evaluates a trained model on n new instances with ILDS(1), ILDS(2), ILDS with given budget and DFS
# Any heuristic using specific state representation should have "specific" in its file name
#include("../common/experiment.jl")
include("../comparison/comparison.jl")
include("utils.jl")

#include("experiment.jl")
#include("../comparison/comparison.jl")
#include("utils.jl")
using CUDA

function benchmark(folder::String, n::Int, chosen_features, has_objective::Bool, generator, basicHeuristics, include_dfs, budget::Int; verbose=true, ILDS = nothing)
    println("Computing benchmarks...")
    models=[]
    models_names=[]
    for file in readdir(folder)
        if splitext(file)[2] == ".bson"
            @load folder * "/" * file model
            push!(models, model)
            push!(models_names, replace(splitext(file)[1], "model_"=>""))
        end
    end

    reward = SeaPearl.GeneralReward

    eval_strategies = SeaPearl.SearchStrategy[]
    search_strategy_names = String[]
    budget_for_strat = []
    if typeof(ILDS) == SeaPearl.ILDSearch
        for i in 0:ILDS.d
            push!(eval_strategies, SeaPearl.ILDSearch(i))
            push!(search_strategy_names, "ILDS"*string(i))
            push!(budget_for_strat, nothing)
        end 
        expo = 2
        while 10^expo <= budget
            push!(eval_strategies, SeaPearl.ILDSearch(10))
            push!(search_strategy_names, "ILDSearch"*string(10^expo))
            push!(budget_for_strat, 10^expo)
            push!(eval_strategies, SeaPearl.DFSearch())
            push!(search_strategy_names, "DFSearch"*string(10^expo))
            push!(budget_for_strat, 10^expo)
            expo = expo + 1
        end
    else
        eval_strategies = SeaPearl.SearchStrategy[SeaPearl.ILDSearch(0),SeaPearl.ILDSearch(1),SeaPearl.ILDSearch(2)]
        search_strategy_names = ["ILDS0", "ILDS1", "ILDS2"]
        budget_for_strat = Any[nothing, nothing, nothing] 
        expo = 2
        while 10^expo <= budget
            push!(eval_strategies, SeaPearl.ILDSearch(10))
            push!(search_strategy_names, "ILDSearch"*string(10^expo))
            push!(budget_for_strat, 10^expo)
            push!(eval_strategies, SeaPearl.DFSearch())
            push!(search_strategy_names, "DFSearch"*string(10^expo))
            push!(budget_for_strat, 10^expo)
            expo = expo + 1
        end
    end
    if include_dfs
        push!(eval_strategies,SeaPearl.DFSearch())
        push!(search_strategy_names, "DFS")
        push!(budget_for_strat, nothing)
    end

    agents = []
    valueSelectionArray = SeaPearl.ValueSelection[]
    
    for (i,model) in enumerate(models)
        agent = RL.Agent(
            policy=RL.QBasedPolicy(
                learner=RL.DQNLearner(
                    approximator=RL.NeuralNetworkApproximator(
                        model=model,
                        optimizer=ADAM()
                    ) |> cpu,
                    target_approximator=RL.NeuralNetworkApproximator(
                        model=model,
                        optimizer=ADAM()
                    ) |> cpu,
                    loss_func=Flux.Losses.huber_loss
                ),
                explorer=RL.EpsilonGreedyExplorer(
                    ϵ_stable=0.0
                )
            ),
            trajectory=RL.CircularArraySLARTTrajectory(
                capacity=1,
                state=SeaPearl.DefaultTrajectoryState[] => (),
                legal_actions_mask=Vector{Bool} => (1,),
            )
            )
        push!(agents,agent)
        if occursin("specific",models_names[i])
            if isa(generator, SeaPearl.MaximumIndependentSetGenerator) 
                state_representation = SeaPearl.MISStateRepresentation{SeaPearl.MISFeaturization,SeaPearl.DefaultTrajectoryState}
            elseif isa(generator, SeaPearl.LegacyGraphColoringGenerator) || isa(generator, SeaPearl.HomogeneousGraphColoringGenerator) || isa(generator, SeaPearl.ClusterizedGraphColoringGenerator) || isa(generator, SeaPearl.BarabasiAlbertGraphGenerator) || isa(generator, SeaPearl.ErdosRenyiGraphGenerator) || isa(generator, SeaPearl.WattsStrogatzGraphGenerator)
                state_representation = SeaPearl.GraphColoringStateRepresentation{SeaPearl.GraphColoringFeaturization, SeaPearl.DefaultTrajectoryState}
            end
            push!(valueSelectionArray, SeaPearl.SimpleLearnedHeuristic{state_representation,reward,SeaPearl.FixedOutput}(agent))
        elseif occursin("homogeneous",models_names[i])
            state_representation = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
            push!(valueSelectionArray, SeaPearl.SimpleLearnedHeuristic{state_representation,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features))
        else
            state_representation = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.HeterogeneousTrajectoryState}
            push!(valueSelectionArray, SeaPearl.SimpleLearnedHeuristic{state_representation,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features))
        end
    end
    append!(valueSelectionArray, collect(values(basicHeuristics)))
    append!(models_names, collect(keys(basicHeuristics)))
    variableHeuristic = SeaPearl.MinDomainVariableSelection{has_objective}()
    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=n)
    
    if !isdir(folder * "/eval_instances/")
        eval_instances_dir = mkdir(folder*"/eval_instances/")
    else
        eval_instances_dir = folder*"/eval_instances/"
    end
    for (idx,instance) in enumerate(evaluator.instances)
        if !isnothing(instance.adhocInfo)
            open(eval_instances_dir*"instance_"*string(idx)*".txt", "w") do io
                write(io, generate_graph_txt(instance.adhocInfo))
            end
        end  
    end
    folder_names = split(folder, "/")
    if !isdir(folder * "/benchmarks")
        dir = mkdir(folder * "/benchmarks")
    else
        dir = folder * "/benchmarks"
    end
    for (j, search_strategy) in enumerate(eval_strategies)
        println("Evaluation with strategy : ", search_strategy)

        SeaPearl.setNodesBudget!(evaluator, budget_for_strat[j])

        SeaPearl.evaluate(evaluator, variableHeuristic, search_strategy; verbose = verbose)
        eval_metrics = evaluator.metrics
        for i in 1:size(eval_metrics)[2]
            SeaPearlExtras.storedata(eval_metrics[:, i]; filename= dir *"/"* search_strategy_names[j] * "_" * models_names[i])
        end
        # empty evaluator metrics
        SeaPearl.resetNodesBudget!(evaluator)

        empty!(evaluator)
    end
end