#!/bin/bash

MAXNUM=$1
F_NAME=$2
F_TYPE=$3
SCRIPT_DIR=$(dirname $0)

exec > missing.txt
for ((i=0 ; i<=${MAXNUM} ; i++)) ; do
   if [ ! -f "${SCRIPT_DIR%/}/${F_NAME}_$i.${F_TYPE}" ] ; then
       echo $i
   fi
done