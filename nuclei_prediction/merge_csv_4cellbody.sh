#!/bin/bash

cd "$1"
# cd /n/groups/htem/users/skuroda/nuclei_output
awk '(NR == 1) || (FNR > 1)' cellbody_cord_id_*.csv > merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' merged.csv > merged2.csv # remove completely same info
sed '/0,0,0,0/d' merged2.csv > merged3.csv #remove all zero rows. 
awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' merged3.csv > info_cellbody.csv # remove rows with same segIDs (column4), but leave segID=0

cp "$1"/info_cellbody.csv ~

sed '/0,0,0,0/d' merged2.csv > merged3.csv #remove all zero rows. 

awk -F ',' 'NR==1 || int($4) == 0 ' info_cellbody.csv > seg0.csv # select segID=0

# new_nuc
awk '(NR == 1) || (FNR > 1)' new_nuc_*.csv > newmerged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' newmerged.csv > newmerged2.csv # remove completely same info
sed '/0,0,0,0/d' newmerged2.csv > newmerged3.csv #remove all zero rows. 
sed '1d' newmerged3.csv > newmerged4.csv
cat info_cellbody.csv newmerged4.csv > newmerged5.csv
awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' newmerged5.csv > info_cellbody_new.csv # remove rows with same segIDs (column4), but leave segID=0

awk -F ',' 'NR==1 || $4 >= 1 { print } ' info_cellbody_new.csv > info_cellbody_20210721.csv

$4 >= 1 { print }

awk '(NR == 1) || (FNR > 1)' cellbody_cord_id_*.csv > merged.csv # leave first line (header) and merge everything else
awk -F ',' 'NR==1 || !seen[$0]++' merged.csv > merged2.csv # remove completely same info
sed '/0,0,0,0/d' merged2.csv > merged3.csv #remove all zero rows. 
awk -F ',' 'NR==1 || int($4) == 0 || !seen[$4]++' merged3.csv > info_cellbody.csv # remo


paste -d , ncount_*.csv > ncount_merged.csv

ls -1 ncount_*.csv | split -l 1000 -d - lists
for list in lists*; do paste -d , $(cat $list) > merge${list##lists}; done
paste -d , merge* > ncount_merged.csv