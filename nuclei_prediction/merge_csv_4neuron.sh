#!/bin/bash

cd "$1"
# cd ../Output
awk '(NR == 1) || (FNR > 1)' cellbody_and_neuron_*.csv > neuron_merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' neuron_merged.csv > neuron_merged2.csv # remove completely same info
awk -F ',' 'NR==1 || {s=0; for (i=1;i<=NF;i++) s+=$i; if (s!=0) print}' neuron_merged2.csv > info_neuron.csv #remove all zero rows. 
cp "$1"/info_neuron.csv ~