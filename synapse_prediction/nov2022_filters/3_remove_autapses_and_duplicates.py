#!/usr/bin/env python3

# Total number of links after score thresholding
n = 56514601

links_to_remove = set()
def load_links(filename):
    with open(filename, 'r') as f:
        line = f.readline()  # Skip header
        line = f.readline()
        while line:
            line = line.strip().split(',')
            links_to_remove.add(tuple([int(i) for i in line[:6]]))
            line = f.readline()
load_links('20221109_fanc_synapses_filtered_zero_sv_scoresover12_autapses.csv')
load_links('20221109_fanc_synapses_filtered_zero_sv_scoresover12_duplicatestoremove.csv')
    

# Load all links
input_table = '20221109_fanc_synapses_filtered_zero_sv_scoresover12.csv'
with open(input_table, 'r') as inf, open(input_table.replace('.csv', '_noautapses_noduplicates.csv'), 'w') as outf:
    outf.write(inf.readline())  # Header
    for i in range(n):
        if i % 100_000 == 0: print(i, '/', n)
        line = inf.readline()
        if tuple([int(s) for s in line.strip().split(',')[1:7]]) not in links_to_remove:
            outf.write(line)
