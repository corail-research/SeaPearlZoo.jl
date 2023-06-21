import ArgParse.ArgParseSettings
import ArgParse.@add_arg_table
import ArgParse.parse_args
using CSV, DataFrames
using Random

# using Flux
# using LightGraphs
# using Random
# using ReinforcementLearning
# const RL = ReinforcementLearning
# using SeaPearl


# include("coloring_config.jl")
# include("coloring_models.jl")
# include("coloring_pipeline.jl")
# include("graph_coloring.jl")

function parse_commandline()
    """
    Parse the command line arguments and return a dictionary containing the values
    """
    s = ArgParseSettings()

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
            default = 100
            required = false
        "--csv_path"
            help = "name of the csv file path for saving performance, if not found, nothing is saved"
            arg_type = String
            required = false
    end
    return parse_args(s)
end

function set_settings()
    """
    Main function of the script
    """
    parsed_args = parse_commandline()

    random_seed = parsed_args["random_seed"]
    time_limit = parsed_args["time_limit"]
    memory_limit = parsed_args["memory_limit"]
    nb_core = parsed_args["nb_core"]
    nb_episodes = parsed_args["nbEpisodes"]
    csv_path = parsed_args["csv_path"]

    eval_freq = ceil(nb_episodes/10)

    if isnothing(csv_path)
        csv_path = ""
        save_performance = false
    else
        save_performance = true
    end

    @eval(Base.Sys, CPU_THREADS=$nb_core)

    Random.seed!(random_seed)

    coloring_settings = ColoringExperimentSettings(nb_episodes, 1, eval_freq, 50, 1, 20, 5, 0.95)
    instance_generator = SeaPearl.BarabasiAlbertGraphGenerator(coloring_settings.nbNodes, coloring_settings.nbMinColor)

    return coloring_settings, instance_generator, csv_path
end


# metricsArray, eval_metricsArray = run_graph_coloring(nb_episodes, eval_freq)
# println("Done ! ")
    # save_csv(metricsArray, eval_metricsArray, csv_path, save_performance)
# end

# main()