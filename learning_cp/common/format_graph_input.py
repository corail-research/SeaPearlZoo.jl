#path = "/home/martom/SeaPearl/SeaPearlZoo/learning_cp/comparison/2022-10-03/exp_thanos_024_GvsS_General_MC_50_50_50_10000_175_17-12-21/eval_instances/"


"""
    This python script is usefull to change the format of the input graph given as a .txt file.  
"""

with open(path+'my_input.txt') as f:
    lines = f.readlines()
    
output = []
output.append(lines[0])
for i,line in enumerate(lines[1:]) :
    print(i, line.split())
    for j in line.split() :
        print(j)
        if int(j) > i+1 :
            print(str(i+1)+" "+j+" 1\n")
            output.append(str(i+1)+" "+j+" 1\n")

with open(path+'my_output.txt', 'w') as f:
    f.write(" ".join(output))