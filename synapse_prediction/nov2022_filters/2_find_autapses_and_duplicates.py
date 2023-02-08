#!/usr/bin/env python3
"""
Find autapses and find pairs of supervoxels that
are connected by multiple synaptic links.
This requires keeping the whole table in memory so
will use a bit over ~32GB RAM.
"""

def to_int(l):
    return [int(i) for i in l]

# Total number of links after thresholding
n = 56514601

# Load links from Stephan's table
sg_table = '20221109_fanc_synapses_filtered_zero_sv_scoresover12.csv'
svid_counts = dict()
with open(sg_table, 'r') as f:
    header = f.readline()
    for i in range(n):
        if i % 100_000 == 0: print(i, '/', n)
        line = f.readline().strip().split(',')
        # Columns 1-3 are pre-xyz, columns 4-6 are post-xyz, columns 8-9 are svids
        link = to_int(line[1:7])+[float(line[7])]
        try:
            svid_counts[tuple(to_int(line[8:10]))].append(link)
        except KeyError:
            svid_counts[tuple(to_int(line[8:10]))] = [link]

# Find autapses (links that connect a given SV to itself)
autapses = {k: v for k, v in svid_counts.items() if k[0] == k[1]}
with open('20221109_fanc_synapses_filtered_zero_sv_scoresover12_autapses.csv', 'w') as f:
    f.write('pre_x,pre_y,pre_z,post_x,post_y,post_z,score,pre_sv_id,post_sv_id\n')
    for k, v in autapses.items():
        for link in v:
            f.write('{},{},{},{},{},{},{:.2f},'.format(*link) + '{},{}\n'.format(*k))
autapses_counts = [len(v) for v in autapses.values()]
print(f'There are {sum(autapses_counts)} autapses')


# Find duplicates (multiple links that connect the same pair of SVs multiple times)
duplicates = {k: v for k, v in svid_counts.items() if k[0] != k[1] and len(v) > 1}
with open('20221109_fanc_synapses_filtered_zero_sv_scoresover12_duplicates.csv', 'w') as f:
    f.write('pre_x,pre_y,pre_z,post_x,post_y,post_z,score,pre_sv_id,post_sv_id\n')
    for k, v in duplicates.items():
        for link in v:
            f.write('{},{},{},{},{},{},{:.2f},'.format(*link) + '{},{}\n'.format(*k))
duplicate_counts = [len(v) for v in duplicates.values()]
#print(f'Removing duplicates would remove {sum(duplicate_counts) - len(duplicate_counts)} synapses')

# For each svid pair, remove the link with the highest score so that that one isn't considered a duplicate
for links in duplicates.values():
    links.remove(max(links, key=lambda x: x[-1]))
with open('20221109_fanc_synapses_filtered_zero_sv_scoresover12_duplicatestoremove.csv', 'w') as f:
    f.write('pre_x,pre_y,pre_z,post_x,post_y,post_z,score,pre_sv_id,post_sv_id\n')
    for k, v in duplicates.items():
        for link in v:
            f.write('{},{},{},{},{},{},{:.2f},'.format(*link) + '{},{}\n'.format(*k))
print(f'Removing duplicates would remove {sum([len(v) for v in duplicates.values()])} synapses')
