#!/usr/bin/env python3
"""
Load a synapse table from .csv, keep only ones with score > 12,
and save the results to a new .csv file
"""

# Stephan Gerhard provided the starting .csv file. He generated this file by
# downloading synaptic links from google cloud and removing any that had
# either their presynaptic or postsynaptic supervoxel ID being 0.
fn = '20221109_fanc_synapses_filtered_zero_sv.csv'
n = 83080692
out_n = 0
with open(fn, 'r') as inf, open(fn.replace('.csv', '_scoresover12.csv'), 'w') as outf:
    outf.write(inf.readline())
    for i in range(n):
        if i % 100_000 == 0: print(i, '/', n)
        line = inf.readline()
        if float(line.strip().split(',')[7]) > 12:
            outf.write(line)
            out_n += 1

print('# of synapses with score over 12:', out_n)
