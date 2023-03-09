using SeaPearl
using SeaPearlExtras
using ReinforcementLearning
const RL = ReinforcementLearning
using Flux
using GeometricFlux
using JSON
using BSON: @save, @load
using Dates
using Random
using LightGraphs
using OrderedCollections
using CircularArrayBuffers
using PyCall
using Pkg
using TensorBoardLogger, Logging

"""
ENV["PYTHON"] = "/home/x86_64-unknown-linux_ol7-gnu/anaconda-2022.05/bin/python"
using Pkg
Pkg.build("PyCall")
run(`$(PyCall.python) -m pip install matplotlib`)
run(`$(PyCall.python) -m pip install pandas`)
run(`$(PyCall.python) -m pip install seaborn`)
run(`$(PyCall.python) -m pip install ipython`)
"""

# -------------------
# -------------------
# Core function
# -------------------
# -------------------

py"""
import sys
sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")

from benchmarkPy import *
import plots
import numpy as np
def benchmark(path):
    print_all(path +"/")

def plot(path,eval):
    plots.all(path +"/", window=100, estimator=np.mean, ilds= eval)
"""
function generate_graph_txt(graph)
    str =""
    str *=string(length(graph.fadjlist))*' '*string(graph.ne)*"\n"
    for neighbors in graph.fadjlist
        str *= join(neighbors," ")*"\n"
    end
    return str
end

dir = "/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/experiment_for_paper/Final_Benchmark/MIS/S/merged_result/" #don't forget to add the "/"
#for other experiment
"""
chosen_features = Dict(
    "variable_is_bound" => true,
    "variable_assigned_value" => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size" => true,
    "variable_is_objective" => true,
    "constraint_activity" => true,
    "constraint_type" => true,
    "nb_not_bounded_variable" => true,
    "values_raw" => true,
)
feature_size = [5, 4, 1]
"""

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
feature_size = [6, 5, 2] 


generator = SeaPearl.MaximumIndependentSetGenerator(30, 6)
#generator = SeaPearl.ClusterizedGraphColoringGenerator(80,12, 0.9)
#generator = SeaPearl.MaxCutGenerator(50,4)

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

n = 20 # Number of instances to evaluate on
budget = 10000 # Budget of visited nodes
has_objective = false # Set it to true if we have to branch on the object ive variable
eval_strategy = SeaPearl.DFSearch()
include_dfs = (eval_strategy == SeaPearl.DFSearch()) # Set it to true if you want to evaluate with DFS in addition to ILDS

selectMin(x::SeaPearl.IntVar; cpmodel=nothing) = SeaPearl.minimum(x.domain)
heuristic_min = SeaPearl.BasicHeuristic(selectMin)
basicHeuristics = OrderedDict()
for i in 1:10
    push!(basicHeuristics,"random"*string(i) => SeaPearl.RandomHeuristic())
end
#push!(basicHeuristics,"min" => heuristic_min)
push!(basicHeuristics,"impact" => SeaPearl.ImpactHeuristic())
push!(basicHeuristics,"activity" => SeaPearl.ActivityHeuristic())

include("../common/benchmark.jl")
include("../common/performanceprofile.jl")
Base.invokelatest(benchmark, dir, n, chosen_features, has_objective, generator, basicHeuristics, include_dfs, budget; ILDS = eval_strategy)
#Base.invokelatest(performanceProfile, dir, n, chosen_features, has_objective, generator, basicHeuristics, budget; ILDS = eval_strategy)

py"""
import sys
sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
from benchmarkPy import *
import plots
import numpy as np
def benchmark(path):
    print_all(path +"/")

def plot(path,eval):
    plots.all(path +"/", window=100, estimator=np.mean, ilds= eval)

def performance(path):
    plots.performance(path +"/", ilds= eval)
"""
dir = replace(dir,"/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/" => "" )

println(dir)
py"benchmark"(dir)
#py"performance"(dir)

"""
#add module load anaconda
import sys
sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
from benchmarkPy import *
import plots
import numpy as np
#path ="experiment_for_paper/Final_Benchmark/MIS/M/exp_010_thanos0_MIS_final_benchmark_S_50_50_10000_161_15-47-33"
#plots.performance(path +"/", ilds= eval)

path="experiment_for_paper/Final_Benchmark/Max-Cut/M/exp_maxwell05_020_final_benchmark_MC_M_50_50_10000_161_17-08-48/"
print_all(path)
"""

