include("../common/experiment.jl")

# -------------------
# Parameters
# -------------------

NB_EPISODES = @isdefined(NB_EPISODES) ? NB_EPISODES : 1001
EVAL_FREQ = @isdefined(EVAL_FREQ) ? EVAL_FREQ : 200
NB_INSTANCES = @isdefined(NB_INSTANCES) ? NB_INSTANCES : 10
NB_RANDOM_HEURISTICS = @isdefined(NB_RANDOM_HEURISTICS) ? NB_RANDOM_HEURISTICS : 0
RESTART_PER_INSTANCES = @isdefined(RESTART_PER_INSTANCES) ? RESTART_PER_INSTANCES : 1
VERBOSE = @isdefined(VERBOSE) ? VERBOSE : false

# -------------------
# Generator
# -------------------
nbNodes = @isdefined(SIZE) ? SIZE : 10
nbMinColor = 5
density = 0.95

OUTPUT_SIZE = nbNodes

coloring_generator = SeaPearl.ClusterizedGraphColoringGenerator(nbNodes, nbMinColor, density)

# -------------------
# State Representation
# -------------------
SR_default_default = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
SR_default_chosen = SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.DefaultTrajectoryState}
SR_heterogeneous = SeaPearl.HeterogeneousStateRepresentation{SeaPearl.DefaultFeaturization,SeaPearl.HeterogeneousTrajectoryState}

numInFeaturesDefault = 3
numInFeaturesDefaultChosen = 6 + nbNodes
numInFeaturesHeterogeneous = [1, 2, nbNodes]

# -------------------
# Agent definition
# -------------------
include("../common/agents.jl")

# -------------------
# Value Heuristic definition
# -------------------

chosen_features = Dict(
    "constraint_activity" => false,
    "constraint_type" => true,
    "nb_involved_constraint_propagation" => false,
    "nb_not_bounded_variable" => false,
    "variable_domain_size" => false,
    "variable_initial_domain_size" => true,
    "variable_is_bound" => false,
    "values_onehot" => true,
    "values_raw" => false,
)

# Learned Heuristic
learnedHeuristic_default_default = SeaPearl.SimpleLearnedHeuristic{SR_default_default,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_default)
learnedHeuristic_default_chosen = SeaPearl.SimpleLearnedHeuristic{SR_default_chosen,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_default_chosen; chosen_features=chosen_features)
learnedHeuristic_heterogeneous = SeaPearl.SimpleLearnedHeuristic{SR_heterogeneous,SeaPearl.GeneralReward,SeaPearl.FixedOutput}(agent_heterogeneous; chosen_features=chosen_features)

# Basic value-selection heuristic
selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)

learnedHeuristics = OrderedDict(
    "defaultdefault" => learnedHeuristic_default_default,
    "defaultchosen" => learnedHeuristic_default_chosen,
    "heterogeneous" => learnedHeuristic_heterogeneous
)
basicHeuristics = OrderedDict(
    "min" => heuristic_min
)

# -------------------
# Variable Heuristic definition
# -------------------
variableHeuristic = SeaPearl.MinDomainVariableSelection{false}()

# -------------------
# Run Experiment
# -------------------

expParameters = Dict(
    :generatorParameters => Dict(
        :nbNodes => nbNodes,
        :nbMinColor => nbMinColor,
        :density => density
    ),
)

metricsArray, eval_metricsArray = trytrain(
    nbEpisodes=NB_EPISODES,
    evalFreq=EVAL_FREQ,
    nbInstances=NB_INSTANCES,
    restartPerInstances=RESTART_PER_INSTANCES,
    generator=coloring_generator,
    variableHeuristic=variableHeuristic,
    learnedHeuristics=learnedHeuristics,
    basicHeuristics=basicHeuristics;
    out_solver=true,
    verbose=VERBOSE,
    expParameters=expParameters,
    nbRandomHeuristics=NB_RANDOM_HEURISTICS,
    exp_name = string(NB_EPISODES) * "_" * string(nbNodes) * "_"
)
nothing
