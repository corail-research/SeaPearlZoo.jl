import ArgParse.ArgParseSettings
import ArgParse.@add_arg_table
import ArgParse.parse_args
using CSV, DataFrames
using Random

function parse_commandline()
    """
    Parse the command line arguments and return a dictionary containing the values
    """
    s = ArgParseSettings()
    choices = ["cpu", "gpu"]

    @add_arg_table s begin
        "--random_seed", "-s"
            help = "seed to intialize the random number generator"
            arg_type = Int
            default = 0
            required = false
        "--time_limit", "-t"
            help = "total CPU time (in seconds) allowed"
            arg_type = Int
            default = 1000000
            required = false
        "--memory_limit", "-m"
            help = "total amount of memory (in MiB) allowed"
            arg_type = Int
            default = 1000000
            required = false
        "--nb_core", "-c"
            help = "number of processing units allocated"
            arg_type = Int
            default = Base.Sys.CPU_THREADS
            required = false
        "--nbEpisodes", "-e"
            help = "number of episodes of training the agent"
            arg_type = Int
            default = 200
            required = false
        "--evalFreq"
            help = "frequence for the evaluation"
            arg_type = Int
            default = 20
            required = false
        "--evalTimeOut"
            help = "time out for the evaluation"
            arg_type = Int
            default = 60
            required = false
        "--nbNodes"
            help = "number of nodes of the graph instances for training"
            arg_type = Int
            default = 10
            required = false
        "--nbNodesEval"
            help = "number of nodes of the graph instances for evaluation"
            arg_type = Int
            default = 10
            required = false
        "--seedEval"
            help = "seed for evaluation"
            arg_type = Int
            default = 123
            required = false
        "--restartPerInstances"
            help = "number of restart per instance"
            arg_type = Int
            default = 10
            required = false
        "--nbInstances"
            help = "number of instances"
            arg_type = Int
            default = 20
            required = false
        "--nbRandomHeuristics"
            help = "number of random heuristics"
            arg_type = Int
            default = 1
            required = false
        "--nbMinColor"
            help = "minimum number of colors in the generated graphs"
            arg_type = Int
            default = 5
            required = false
        "--density"
            help = "density of the generated graphs"
            arg_type = Float64
            default = 0.95
            required = false
        "--save_model"
            help = "save the model"
            arg_type = Bool
            default = false
            required = false
        "--csv_path"
            help = "name of the csv file path for saving performance, if not found, nothing is saved"
            arg_type = String
            required = false
        "--device"
            help = "device to use for training (cpu or gpu)"
            arg_type = String
            default = "cpu"
            required = false
            range_tester = (x->x ∈ choices)
        "--path_json", "-j"
            help = "use a json file to set the parameters"
            arg_type = String
            default = nothing
            required = false
    end
    return parse_args(s)
end

function set_settings()
    """
    Main function of the script
    """
    parsed_args = parse_commandline()

    if !isnothing(parsed_args["path_json"])
        parameters = SeaPearl.read_parameters(parsed_args["path_json"])

        random_seed = parameters["random_seed"]
        time_limit = parameters["time_limit"]
        memory_limit = parameters["memory_limit"]
        nb_core = parameters["nb_core"]
        nb_episodes = parameters["nbEpisodes"]
        nb_nodes = parameters["nbNodes"]
        nb_nodes_eval = parameters["nbNodesEval"]
        restart_per_instances = parameters["restartPerInstances"]
        nb_instances = parameters["nbInstances"]
        nb_random_heuristics = parameters["nbRandomHeuristics"]
        nb_min_color = parameters["nbMinColor"]
        density = parameters["density"]
        eval_timeout = parameters["evalTimeOut"]
        eval_freq = parameters["evalFreq"]
        seedEval = parameters["seedEval"]
        save_model = parameters["save_model"]
        csv_path = parameters["csv_path"]
        device = parameters["device"]
    else
        random_seed = parsed_args["random_seed"]
        time_limit = parsed_args["time_limit"]
        memory_limit = parsed_args["memory_limit"]
        nb_core = parsed_args["nb_core"]
        nb_episodes = parsed_args["nbEpisodes"]
        nb_nodes = parsed_args["nbNodes"]
        nb_nodes_eval = parsed_args["nbNodesEval"]
        restart_per_instances = parsed_args["restartPerInstances"]
        nb_instances = parsed_args["nbInstances"]
        nb_random_heuristics = parsed_args["nbRandomHeuristics"]
        nb_min_color = parsed_args["nbMinColor"]
        density = parsed_args["density"]
        eval_timeout = parsed_args["evalTimeOut"]
        eval_freq = parsed_args["evalFreq"]
        seedEval = parsed_args["seedEval"]
        save_model = parsed_args["save_model"]
        csv_path = parsed_args["csv_path"]
        device = parsed_args["device"]
    end

    if isnothing(csv_path)
        csv_path = ""
        save_performance = false
    else
        save_performance = true
    end

    # @eval(Base.Sys, CPU_THREADS=$nb_core)

    Random.seed!(random_seed)

    coloring_settings = ColoringExperimentSettings(nb_episodes, restart_per_instances, eval_freq, eval_timeout, seedEval, nb_instances, nb_random_heuristics, nb_nodes, nb_nodes_eval, nb_min_color, density)

    instance_generator = SeaPearl.ClusterizedGraphColoringGenerator(coloring_settings.nbNodes, coloring_settings.nbMinColor, coloring_settings.density)
    eval_generator = SeaPearl.ClusterizedGraphColoringGenerator(coloring_settings.nbNodesEval, coloring_settings.nbMinColor, coloring_settings.density)
    
    return coloring_settings, instance_generator, eval_generator, csv_path, save_model, device
end