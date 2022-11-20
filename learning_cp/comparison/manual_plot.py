import os
import re
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.ticker as ticker
pd.options.display.float_format = '{:,.4f}'.format

sys.path.insert(0, "/home/martom/SeaPearl/SeaPearlExtras.jl/src/metrics/basicmetrics/")
from benchmarkPy import *
from plots import *

path ="/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/experiment_for_paper/Score vs General/Merged_results/GC50/"
eval = get_eval(path)
training = get_training(path)

save_path=None
if save_path is None:
    save_path = path

def score_first(eval, estimator=np.mean, ax=None, save_path=None, training=None):
    eval["Episode"] = (eval["Episode"]-1)/max(eval["Episode"])*max(training["Episode"])*10
    first_solution = eval.loc[eval[(eval["SolutionFound"] == 1)].groupby(["Episode", "Instance", "Heuristic"])["Solution"].idxmin()].sort_values("Heuristic", key=lambda series: apply_key(series, training))
    #first_solution.loc[first_solution["Score"] == 0, "Score"] = -88
    last_solution = eval.loc[eval[eval["SolutionFound"] == 1].groupby(["Episode", "Instance", "Heuristic"])["Solution"].idxmax()].sort_values("Heuristic", key=lambda series: apply_key(series, training))
    #first_solution = first_solution.loc[first_solution["Episode"] <= 75]
    #last_solution = last_solution.loc[last_solution["Episode"] <= 75]
    data_opti = last_solution.groupby(["Instance","Episode"])["Score"].min()
    data_opti = data_opti.to_frame().reset_index()
    data_opti= data_opti.assign(Heuristic = "Optimal Score")
    data_opti.loc[data_opti["Instance"] == 1, "Score"] = 9
    data_opti.loc[data_opti["Instance"] == 2, "Score"] = 9
    data_opti.loc[data_opti["Instance"] == 3, "Score"] = 8
    data_opti.loc[data_opti["Instance"] == 4, "Score"] = 8
    data_opti.loc[data_opti["Instance"] == 5, "Score"] = 8
    data_opti.loc[data_opti["Instance"] == 6, "Score"] = 8 
    data_opti.loc[data_opti["Instance"] == 7, "Score"] = 8 
    data_opti.loc[data_opti["Instance"] == 8, "Score"] = 8 
    data_opti.loc[data_opti["Instance"] == 9, "Score"] = 8
    data_opti.loc[data_opti["Instance"] == 10, "Score"] = 8
    first_solution = pd.concat([first_solution,data_opti]).reset_index()
    plot = sns.lineplot(data=first_solution, y="Score", x="Episode", hue="Heuristic", estimator=estimator, ax=ax)
    ax.set_title("Score at first dive", fontsize=22, fontweight="medium")
    plot.legend(fontsize='x-large', title_fontsize='40', loc='upper right')
    ax.set_xlabel("Training episode", fontsize=18) 
    ax.set_ylabel("Score obtained", fontsize=18) 
    plot.set_yticklabels(plot.get_yticks(), size = "x-large")
    plot.set_xticklabels(plot.get_xticks(), size = "x-large")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x)))
    save_fig(plot, save_path, "eval_score_first_solution")

def node_best(eval, estimator=np.mean, ax=None, save_path=None, training=None):
    eval["Episode"] = (eval["Episode"]-1)/max(eval["Episode"])*max(training["Episode"])*4
    first_solution = eval.loc[eval[eval["SolutionFound"] == 1].groupby(["Episode", "Instance", "Heuristic"])["Score"].idxmin()].sort_values("Heuristic", key=lambda series: apply_key(series, training))
    idx_DFS = first_solution.loc[(first_solution["Episode"]==0) & (first_solution["Heuristic"]=="DFS"),"Nodes"].index
    idx_RBS = first_solution.loc[(first_solution["Episode"]==0) & (first_solution["Heuristic"]=="RBS"),"Nodes"].index
    new_values = first_solution.loc[(first_solution["Episode"]==0) & (first_solution["Heuristic"]=="Random"),"Nodes"].values
    first_solution.loc[idx_DFS,"Nodes"]=new_values
    first_solution.loc[idx_RBS,"Nodes"]=new_values
    #first_solution = first_solution.loc[first_solution["Episode"] <= 75]
    plot = sns.lineplot(data=first_solution, y="Nodes", x="Episode", hue="Heuristic", estimator=estimator, ax=ax)
    ax.set_title("Node visited until best solution", fontsize=22, fontweight="medium")
    plot.legend(fontsize='x-large', title_fontsize='40')
    ax.set_xlabel("Training episode", fontsize=18) 
    ax.set_ylabel("Nodes visited", fontsize=18) 
    plot.set_yticklabels(plot.get_yticks(), size = "x-large")
    plot.set_xticklabels(plot.get_xticks(), size = "x-large")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x)))
    save_fig(plot, save_path, "eval_node_visited_best_solution")

sns.set(rc={"figure.figsize": (11, 8)})
sns.set_theme(style="whitegrid")
_, ax = plt.subplots()
eval = get_eval(path)
training = get_training(path)
score_first(eval, estimator=np.mean, save_path=save_path, ax=ax, training=training)
_, ax = plt.subplots()
eval = get_eval(path)
training = get_training(path)
node_best(eval, estimator=np.mean, save_path=save_path, ax=ax, training=training)
