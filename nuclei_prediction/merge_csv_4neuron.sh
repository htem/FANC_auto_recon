cd ../Output
awk '(NR == 1) || (FNR > 1)' cellbody_and_neuron_*.csv > merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' merged.csv > merged2.csv # remove completely same info
awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' merged2.csv > info_cellbody.csv # remove rows with same segIDs (column4), but leave segID=0