# using CUDA
using BSON: @load
using SeaPearl 
using Flux 
using ReinforcementLearning
include("save_metrics.jl")


function generate_graph_txt(graph)
    str =""
    str *=string(length(graph.fadjlist))*' '*string(graph.ne)*"\n"
    for neighbors in graph.fadjlist
        str *= join(neighbors," ")*"\n"
    end
    return str
end

function load_models(folder::String)
    println("Computing benchmarks...")
    models=[]
    model_names=[]
    for file in readdir(folder)
        if splitext(file)[2] == ".bson"
            println(folder * "/" * file)
            @load folder * "/" * file model
            push!(models, model)
            push!(model_names, replace(splitext(file)[1], "model_"=>""))
        end
    end
    return models, model_names
end

function set_strategies(include_dfs, budget, ILDS=nothing)
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
    end
    if include_dfs
        push!(eval_strategies,SeaPearl.DFSearch())
        push!(search_strategy_names, "DFS")
        push!(budget_for_strat, nothing)
    end
    return eval_strategies, search_strategy_names, budget_for_strat

end

function set_agents_and_value_selection(models, model_names, reward, chosen_features, generator, basicHeuristics)
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
        push!(agents, agent)
        state_representation = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.HeterogeneousTrajectoryState}
        push!(valueSelectionArray, SeaPearl.SimpleLearnedHeuristic{state_representation,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features))
    end
    append!(valueSelectionArray, collect(values(basicHeuristics)))
    append!(model_names, collect(keys(basicHeuristics)))
    return valueSelectionArray, model_names
end 

function manage_evaluation_folder(folder, evaluator)
    if !isdir(folder * "/eval_instances/")
        eval_instances_dir = mkdir(folder*"/eval_instances/")
    else
        eval_instances_dir = folder*"/eval_instances/"
    end
    for (idx, instance) in enumerate(evaluator.instances)
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
    return dir
end

function benchmark(;
        models=nothing, 
        model_folder=nothing, 
        evaluation_folder=nothing, 
        num_instances::Int, 
        chosen_features, 
        take_objective::Bool, 
        generator, 
        basicHeuristics, 
        include_dfs, 
        budget::Int, 
        verbose=true, 
        ILDS=nothing,
        save_experiment_metrics=true
    )
    if isnothing(model_folder)
        model_names = ["model_"*string(i) for i in 1:length(models)]
    else
        models, model_names = load_models(model_folder)
    end
    reward = SeaPearl.GeneralReward
    eval_strategies, search_strategy_names, budget_for_strat = set_strategies(include_dfs, budget, ILDS)
    valueSelectionArray, model_names = set_agents_and_value_selection(models, model_names, reward, chosen_features, generator, basicHeuristics)
    variableHeuristic = SeaPearl.MinDomainVariableSelection{take_objective}()
    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=num_instances)
    if !isnothing(model_folder)
        dir = manage_evaluation_folder(model_folder, evaluator)
    end
    if !save_experiment_metrics
        df = DataFrame()
    end
    for (j, search_strategy) in enumerate(eval_strategies)
        println("Evaluation with strategy : ", search_strategy)
        SeaPearl.setNodesBudget!(evaluator, budget_for_strat[j])
        SeaPearl.evaluate(evaluator, variableHeuristic, search_strategy; verbose = verbose)
        eval_metrics = evaluator.metrics
        if save_experiment_metrics
            save_metrics(eval_metrics, "test_benchmark.csv")
        else
            new_df = get_metrics_dataframe(eval_metrics)
            df = vcat(df, new_df)
        end
        SeaPearl.resetNodesBudget!(evaluator)
        empty!(evaluator)
    end
    if !save_experiment_metrics
        return df
    end
end