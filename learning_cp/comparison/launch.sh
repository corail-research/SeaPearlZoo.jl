#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
id=$1
machine="maxwell0"
set -e
for i in {1..8}
do
echo
echo ssh to martom@"$machine$i"
result=$(ssh martom@"$machine$i" "~/SeaPearl/SeaPearlZoo/learning_cp/comparison/./local_launch.sh $id $machine$i "&)
echo $result
if [[ $result  == *'occupied'* ]]; then
echo GPU occupied. Switching to next device.
elif [[ $result  == *'free'* ]]; then
echo Process started on "$machine$i" with id $id.
exit
fi
done
echo "Error, all maxwell01 to maxwell08 GPU occupied."
exit 1
