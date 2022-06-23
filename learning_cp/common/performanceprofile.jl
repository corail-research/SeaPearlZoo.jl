# This util evaluates a trained model on n new instances
include("experiment.jl")
include("../comparison/comparison.jl")
include("utils.jl")

# Parameters to edit
model_file = "../comparison/exp_jobshop_3_10_502022-06-21T12-25-30/model_learning.bson"
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
n = 100
has_objective = true
eval_strategy = SeaPearl.ILDSearch(2)


function performanceProfile(model_file::String, n::Int, chosen_features, has_objective::Bool, generator; eval_strategy = SeaPearl.DFSearch(), verbose=true)
    @load model_file model

    reward = SeaPearl.GeneralReward

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
    SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
    valueSelectionArray = [SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,reward,SeaPearl.FixedOutput}(agent; chosen_features=chosen_features)]
    variableHeuristic = SeaPearl.MinDomainVariableSelection{has_objective}()
    evaluator = SeaPearl.SameInstancesEvaluator(valueSelectionArray, generator; nbInstances=n, evalTimeOut=20)
    SeaPearl.evaluate(evaluator, variableHeuristic, eval_strategy; verbose = verbose)
    eval_metrics = evaluator.metrics[:,1]
    SeaPearlExtras.storedata(eval_metrics; filename= "../performanceprofiles/performanceprofile_" * replace(last(split(model_file,"/")),".bson"=>""))
end

performanceProfile(model_file, n, chosen_features, has_objective, generator; eval_strategy=eval_strategy)

nothing