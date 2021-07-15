cd ../Output
awk '(NR == 1) || (FNR > 1)' cellbody_and_neuron_*.csv > neuron_merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' neuron_merged.csv > info_neuron.csv # remove completely same info
# awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' info_neuron.csv > info_neuron2.csv # remove rows with same segIDs (column4), but leave segID=0