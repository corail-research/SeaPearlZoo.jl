import sys
sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
import benchmarkPy
import plots
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from IPython.display import display
pd.options.display.float_format = '{:,.4f}'.format

def modify_and_import(module_name, package, modification_func):
    spec = importlib.util.find_spec(module_name, package)
    source = spec.loader.get_source(module_name)
    new_source = modification_func(source)
    module = importlib.util.module_from_spec(spec)
    codeobj = compile(new_source, module.__spec__.origin, 'exec')
    exec(codeobj, module.__dict__)
    sys.modules[module_name] = module
    return module

#path = "/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/experiment_for_paper/Score vs General/Merged_results/MC50/"
path = "experiment_for_paper/Final_Benchmark/MIS/L/exp_012_maxwell03_MIS_final_benchmark_M_100_100_10000_161_12-23-18_AAA"

plots = modify_and_import("plots", None, lambda src: src.replace("hello", "world"))

#plots.all(path +"/", window=100, estimator=np.mean, ilds= False)
plots.performance(path +"/", ilds= eval)
