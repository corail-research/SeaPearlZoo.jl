# using CUDA
using BSON: @load
using SeaPearl 
using Flux 
using ReinforcementLearning
using SeaPearlExtras

include("src/learning_cp/mis/mis_config.jl")
include("src/learning_cp/mis/mis_model.jl")

include("src/learning_cp/graph_coloring/coloring_config.jl")
include("src/learning_cp/graph_coloring/dqn_hsr/heterogeneous_coloring_model.jl")

include("src/learning_cp/utils/save_metrics.jl")


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
    models_names=[]
    for file in readdir(folder)
        if splitext(file)[2] == ".bson"
            println(folder * "/" * file)
            @load folder * "/" * file model
            push!(models, model)
            push!(models_names, replace(splitext(file)[1], "model_"=>""))
        end
    end
    return models, models_names
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

function set_agents_and_value_selection(models, models_names, reward, chosen_features, generator, basicHeuristics)
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
    append!(models_names, collect(keys(basicHeuristics)))
    return valueSelectionArray, models_names
end 

function manage_evaluation_folder(folder, evaluator)
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
    return dir
end

function benchmark(folder::String, n::Int, chosen_features, has_objective::Bool, generator, basicHeuristics, include_dfs, budget::Int; verbose=true, ILDS = nothing)
    models, models_names = load_models(folder)

    reward = SeaPearl.GeneralReward

    eval_strategies, search_strategy_names, budget_for_strat = set_strategies(include_dfs, budget, ILDS)

    valueSelectionArray, models_names = set_agents_and_value_selection(models, models_names, reward, chosen_features, generator, basicHeuristics)

    variableHeuristic = SeaPearl.MinDomainVariableSelection{has_objective}()

    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=n)
    
    dir = manage_evaluation_folder(folder, evaluator)


    for (j, search_strategy) in enumerate(eval_strategies)
        println("Evaluation with strategy : ", search_strategy)

        SeaPearl.setNodesBudget!(evaluator, budget_for_strat[j])

        SeaPearl.evaluate(evaluator, variableHeuristic, search_strategy; verbose = verbose)
        eval_metrics = evaluator.metrics
        save_metrics(eval_metrics, "test_benchmark_gc.csv")

        SeaPearl.resetNodesBudget!(evaluator)

        empty!(evaluator)
    end
end



dir = "saved_model/gc" # path to the folder with the saved models

chosen_features = Dict(
    "node_number_of_neighbors" => true,
    "constraint_type" => true,
    "constraint_activity" => true,
    "nb_not_bounded_variable" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "variable_is_objective" => true,
    "variable_assigned_value" => true,
    "variable_is_bound" => true,
    "values_raw" => true)

# generator = SeaPearl.MaximumIndependentSetGenerator(30, 6)
generator = SeaPearl.ClusterizedGraphColoringGenerator(80, 12, 0.9)

n = 20 # Number of instances to evaluate on
budget = 10000 # Budget of visited nodes
has_objective = false # Set it to true if we have to branch on the object ive variable
eval_strategy = SeaPearl.DFSearch()
include_dfs = true # Set it to true if you want to evaluate with DFS in addition to ILDS

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
basicHeuristics = Dict()
for i in 1:10
    push!(basicHeuristics,"random"*string(i) => SeaPearl.RandomHeuristic())
end

push!(basicHeuristics,"min" => heuristic_min)

benchmark(dir, n, chosen_features, has_objective, generator, basicHeuristics, include_dfs, budget; ILDS = eval_strategy)

