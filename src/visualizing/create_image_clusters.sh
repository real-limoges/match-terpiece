#!/bin/sh
declare -a norms=("False" "True")
declare -a layers=("fc1" "fc2")

for layer in "${layers[@]}"
do
    for norm in "${norms[@]}"
    do
        for k in {5..19}
        do
            cluster="_cluster.csv"
            CSV="$layer$cluster"
            
            B_MOD="k_"
            UND="_"
            PKL=".pkl"
            MODEL="$B_MOD$k$UND$layer$UND$norm$PKL"

            echo $MODEL            
            python visualize_clusters.py $CSV $MODEL

        done
    done
done
