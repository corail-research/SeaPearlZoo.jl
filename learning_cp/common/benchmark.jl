# This util evaluates a trained model on n new instances with ILDS(1), ILDS(2), ILDS with given budget and DFS
# Any heuristic using specific state representation should have "specific" in its file name
include("experiment.jl")
include("../comparison/comparison.jl")
include("utils.jl")

# Parameters to edit
folder = "../comparison/exp_MIS_50_tripartite_vs_specific_100012022-06-17T11-14-00/"
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

generator = SeaPearl.MaximumIndependentSetGenerator(50, 8)
n = 3
budget = 1000
has_objective = true
include_dfs = true

selectMax(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.maximum(x.domain)
heuristic_max = SeaPearl.BasicHeuristic(selectMax)
basicHeuristics = OrderedDict(
        "max" => heuristic_max,
        "random" => SeaPearl.RandomHeuristic()
)


function benchmark(folder::String, n::Int, chosen_features, has_objective::Bool, generator, basicHeuristics, include_dfs, budget::Int; verbose=true)
    models=[]
    models_names=[]
    for file in readdir(folder)
        if splitext(file)[2] == ".bson"
            @load folder * "/" * file model
            push!(models, model)
            push!(models_names, splitext(file)[1])
        end
    end

    reward = SeaPearl.GeneralReward
    eval_strategies = SeaPearl.SearchStrategy[SeaPearl.ILDSearch(1),SeaPearl.ILDSearch(2),SeaPearl.ILDSearch(10)]
    if include_dfs
        push!(eval_strategies,SeaPearl.DFSearch())
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
                    ),
                    target_approximator=RL.NeuralNetworkApproximator(
                        model=model,
                        optimizer=ADAM()
                    ),
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
        else
            state_representation = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization, SeaPearl.HeterogeneousTrajectoryState}
            push!(valueSelectionArray, SeaPearl.SimpleLearnedHeuristic{state_representation,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features))
        end
    end
    append!(valueSelectionArray, collect(values(basicHeuristics)))
    append!(models_names, collect(keys(basicHeuristics)))
    variableHeuristic = SeaPearl.MinDomainVariableSelection{has_objective}()
    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=n)
    folder_names = split(folder, "/")
    dir = mkdir("../benchmarks/"*folder_names[length(folder_names)-1])
    for search_strategy in eval_strategies
        println(search_strategy)
        if search_strategy == SeaPearl.ILDSearch(10)
            SeaPearl.setNodesBudget!(evaluator, budget)
        end
        SeaPearl.evaluate(evaluator, variableHeuristic, search_strategy; verbose = verbose)
        eval_metrics = evaluator.metrics
        for i in 1:size(eval_metrics)[2]
            SeaPearlExtras.storedata(eval_metrics[:, i]; filename= dir *"/"*string(search_strategy) * "_" * models_names[i])
        end
        # empty evaluator metrics
        if search_strategy == SeaPearl.ILDSearch(10)
            SeaPearl.resetNodesBudget!(evaluator)
        end
        empty!(evaluator)
    end
end

benchmark(folder, n, chosen_features, has_objective, generator, basicHeuristics, include_dfs, budget)

nothing