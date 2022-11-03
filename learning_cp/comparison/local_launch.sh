id=$1
machine=$2
#echo $id 
#echo $machine
echo "$PWD"
gpu=$(nvidia-smi pmon -i 0 -s m -c 1)
if [[ $gpu == *'julia'* ]]; then
echo "GPU occupied."
else
echo "GPU free."
echo $(df /home/martom -H) 
module load julia
cd ~/SeaPearl/SeaPearlZoo/learning_cp/comparison
#nohup julia --project run_tom.jl $machine > output.out
bash -c "exec -a $id nohup julia --project run_tom.jl $machine $id &"
fi
