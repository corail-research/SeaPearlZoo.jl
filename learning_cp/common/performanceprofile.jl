# This util evaluates a trained model on n new instances
include("experiment.jl")
include("../comparison/comparison.jl")
include("utils.jl")

# Parameters to edit
folder = "../comparison/exp_jobshop_3_10_50_heterogeneous_cpnn_10001_30_2022-06-22T17-40-54/"
chosen_features = Dict(
    "node_number_of_neighbors" => true,
    "constraint_type" => true,
    "constraint_activity" => true,
    "nb_not_bounded_variable" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "variable_assigned_value" => true,
    "variable_is_bound" => true,
    "values_raw" => true)
generator = SeaPearl.JobShopGenerator(3, 10, 50)
n = 2
has_objective = true
eval_strategy = SeaPearl.ILDSearch(2)

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
basicHeuristics = OrderedDict(
        "min" => heuristic_min,
        "random" => SeaPearl.RandomHeuristic()
)


function performanceProfile(folder::String, n::Int, chosen_features, has_objective::Bool, generator, basicHeuristics; eval_strategy = SeaPearl.DFSearch(), verbose=true)
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

    agents = []
    for model in models
       push!(agents,RL.Agent(
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
        ))
    end
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    valueSelectionArray = SeaPearl.ValueSelection[SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features) for agent in agents]
    append!(valueSelectionArray, collect(values(basicHeuristics)))
    append!(models_names, collect(keys(basicHeuristics)))
    variableHeuristic = SeaPearl.MinDomainVariableSelection{has_objective}()
    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=n, evalTimeOut=20)
    SeaPearl.evaluate(evaluator, variableHeuristic, eval_strategy; verbose = verbose)
    eval_metrics = evaluator.metrics
    folder_names = split(folder, "/")
    dir = mkdir("../performanceprofiles/"*folder_names[length(folder_names)-1])
    for i in 1:size(eval_metrics)[2]
        SeaPearlExtras.storedata(eval_metrics[:, i]; filename= dir *"/"* models_names[i])
    end
end

performanceProfile(folder, n, chosen_features, has_objective, generator, basicHeuristics; eval_strategy=eval_strategy)

nothing