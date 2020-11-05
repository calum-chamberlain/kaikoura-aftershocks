#!/usr/bin/bash

# Script to loop over kaikoura_detect.py to force closing of python thread to
# enable garbage collection. Required because someone is hogging the memory!

ndays=577
i=87
while [ $i -lt $ndays ]
    do
    python Scripts/kaikoura_detect.py -d $i
    i=$[$i+1]
done
