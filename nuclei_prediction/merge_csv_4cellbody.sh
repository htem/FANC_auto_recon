#!/bin/bash

cd "$1"
# cd /n/groups/htem/users/skuroda/nuclei_output
awk '(NR == 1) || (FNR > 1)' cellbody_cord_id_*.csv > merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' merged.csv > merged2.csv # remove completely same info
awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' merged2.csv > merged3.csv # remove rows with same segIDs (column4), but leave segID=0
sed '/0,0,0,0/d' merged3.csv > info_cellbody.csv #remove all zero rows. 

cp "$1"/info_cellbody.csv ~