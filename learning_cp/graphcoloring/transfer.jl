include("../common/experiment.jl")
include("../common/utils.jl")

#############################
n_nodes = 10
n_min_color = 5
density = 0.30

n_episodes = 3001
n_instances = 10
n_eval = 10
#############################
generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}
chosen_features = Dict(
    "constraint_activity" => true,
    "constraint_type" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "values_raw" => true,
)

agent_heterogeneous = get_heterogeneous_agent(;
    get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=5000, n_actions=n_nodes),
    get_explorer = () -> get_epsilon_greedy_explorer(10000, 0.01),
    batch_size=16,
    update_horizon=8,
    min_replay_history=256,
    update_freq=1,
    target_update_freq=8,
    get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
        feature_size=[2, 3, 1],
        conv_size=8,
        dense_size=16,
        output_size=1,
        n_layers_graph=4,
        n_layers_node=2,
        n_layers_output=2
    )
)
learned_heuristic_heterogeneous = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_heterogeneous; chosen_features=chosen_features)

learnedHeuristics = OrderedDict(
    "heterogeneous" => learned_heuristic_heterogeneous,
)

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

basicHeuristics = OrderedDict(
    "min" => heuristic_min,
    "random" => SeaPearl.RandomHeuristic()
)

variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

expParameters = Dict(
    :generatorParameters => Dict(
        :nbNodes => n_nodes,
        :nbMinColor => n_min_color,
        :density => density
    ),
)

metricsArray, eval_metricsArray = trytrain(
    nbEpisodes=n_episodes,
    evalFreq=Int(floor(n_episodes / n_eval)),
    nbInstances=n_instances,
    restartPerInstances=1,
    generator=generator,
    variableHeuristic=variableHeuristic,
    learnedHeuristics=learnedHeuristics,
    basicHeuristics=basicHeuristics;
    out_solver=true,
    verbose=true,
    expParameters=expParameters,
    nbRandomHeuristics=0,
    exp_name="transfer_" * string(n_episodes) * "_" * string(n_nodes) * "_"
)
nothing

n_nodes = 40
n_episodes = 501
generator = SeaPearl.ClusterizedGraphColoringGenerator(n_nodes, n_min_color, density)

agent_transfer = RL.Agent(
    policy= RL.QBasedPolicy(
        learner=deepcopy(agent_heterogeneous.policy.learner),
        explorer= get_epsilon_greedy_explorer(2000, 0.01),
    ),
    trajectory=get_heterogeneous_slart_trajectory(capacity=5000, n_actions=n_nodes)
)
learned_heuristic_transfer = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_transfer; chosen_features=chosen_features)

agent_heterogeneous = get_heterogeneous_agent(;
get_heterogeneous_trajectory = () -> get_heterogeneous_slart_trajectory(capacity=5000, n_actions=n_nodes),
get_explorer = () -> get_epsilon_greedy_explorer(2000, 0.01),
batch_size=16,
update_horizon=8,
min_replay_history=256,
update_freq=1,
target_update_freq=8,
get_heterogeneous_nn = () -> get_heterogeneous_fullfeaturedcpnn(
    feature_size=[2, 3, 1],
    conv_size=8,
    dense_size=16,
    output_size=1,
    n_layers_graph=4,
    n_layers_node=2,
    n_layers_output=2
)
)
learned_heuristic_heterogeneous = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_heterogeneous; chosen_features=chosen_features)

learnedHeuristics = OrderedDict(
    "heterogeneous" => learned_heuristic_heterogeneous,
    "transfer" => learned_heuristic_transfer,
)

metricsArray, eval_metricsArray = trytrain(
    nbEpisodes=n_episodes,
    evalFreq=Int(floor(n_episodes / n_eval)),
    nbInstances=n_instances,
    restartPerInstances=1,
    generator=generator,
    variableHeuristic=variableHeuristic,
    learnedHeuristics=learnedHeuristics,
    basicHeuristics=basicHeuristics;
    out_solver=true,
    verbose=true,
    expParameters=expParameters,
    nbRandomHeuristics=0,
    exp_name="transfered_" * string(n_episodes) * "_" * string(n_nodes) * "_"
)
nothing
