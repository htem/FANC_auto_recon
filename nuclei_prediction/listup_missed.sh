#!/bin/bash

cd "$1"
# cd ../Output
exec > missing.txt
for ((i=0 ; i<=55187 ; i++)) ; do
   if [ ! -f "cellbody_cord_id_$i.csv" ] ; then
       echo $i
   fi
done
# grep '.....' missing.txt > after10000.txt
cp "$1"/missing.txt ~