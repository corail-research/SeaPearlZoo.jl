
import os
import re
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.display.float_format = '{:,.4f}'.format
import importlib
import sys
sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
import plots
import benchmarkPy

def modify_and_import(module_name, package, modification_func):
    spec = importlib.util.find_spec(module_name, package)
    source = spec.loader.get_source(module_name)
    new_source = modification_func(source)
    module = importlib.util.module_from_spec(spec)
    codeobj = compile(new_source, module.__spec__.origin, 'exec')
    exec(codeobj, module.__dict__)
    sys.modules[module_name] = module
    return module

save_path = None
benchmarkPy = modify_and_import("benchmarkPy", None, lambda src: src.replace("hello", "world"))
plots = modify_and_import("plots", None, lambda src: src.replace("hello", "world"))

path = "/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/experiment_for_paper/Final_Benchmark/GC/L/exp_004_final_benchmark_GC_L_target_500_80_80_6000_161_15-57-06/benchmarks/"
eval = plots.get_eval(path)

save_path = path

sns.set(rc={"figure.figsize": (12, 8)})
sns.set_theme(style="whitegrid")
_, ax = plt.subplots()
plots = modify_and_import("plots", None, lambda src: src.replace("hello", "world"))
plots.performance_plot_score_best(eval, save_path=path, ax=ax)


path = "experiment_for_paper/Final_Benchmark/GC/S/exp_009_maxwell06_final_benchmark_GC_S_20_20_5000_165_00-53-39_better/"
benchmarkPy.print_all(path)
#path = "experiment_for_paper/Final_Benchmark/GC/M/exp_008_maxwell02_final_benchmark_GC_M_40_40_8000_161_00-06-54_AAA/"
#benchmarkPy.print_all(path)
benchmarkPy = modify_and_import("benchmarkPy", None, lambda src: src.replace("hello", "world"))

path = "experiment_for_paper/Final_Benchmark/GC/L/exp_004_final_benchmark_GC_L_target_500_80_80_6000_161_15-57-06/"
benchmarkPy.print_all(path)
path = "experiment_for_paper/Final_Benchmark/Max-Cut/S/exp_maxwell05_016_final_benchmark_MC_S_20_20_10000_165_11-40-33/"
benchmarkPy.print_all(path)
#path = "experiment_for_paper/Final_Benchmark/Max-Cut/M/exp_maxwell05_020_final_benchmark_MC_M_50_50_10000_161_17-08-48/"
#benchmarkPy.print_all(path)
path = "experiment_for_paper/Final_Benchmark/MIS/S/exp_014_thanos00_MIS_final_benchmark_S_sumPooling_30_30_30000_161_00-42-12/"
benchmarkPy.print_all(path)
path = "experiment_for_paper/Final_Benchmark/MIS/M/exp_010_thanos0_MIS_final_benchmark_S_50_50_10000_161_15-47-33/"
benchmarkPy.print_all(path)
path = "experiment_for_paper/Final_Benchmark/MIS/L/exp_012_maxwell03_MIS_final_benchmark_M_100_100_10000_161_12-23-18_AAA/"
benchmarkPy.print_all(path)

