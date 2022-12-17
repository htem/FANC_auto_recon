## Predicting synaptic links with the `synful` package

Convolutional neural networks were trained to predict the locations of postsynaptic sites and their corresponding presynaptic sites in the EM dataset using the method described in [Buhmann et al. 2021 _Nature Methods_](https://www.nature.com/articles/s41592-021-01183-7). The implementation of this pipeline is available on Github at [funkelab/synful](https://github.com/funkelab/synful).

[This neuroglancer link](https://neuromancer-seung-import.appspot.com/#!%7B%22layers%22:%5B%7B%22source%22:%22precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/em/rechunked%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22FANCv4%22%7D%2C%7B%22source%22:%22precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/postsynapses_May2021%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%20emitRGBA%28vec4%281%2C%200%2C%201%2C%20toNormalized%28getDataValue%28%29%29%29%29%3B%20%7D%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22postsynapses_May2021%22%7D%2C%7B%22source%22:%22precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/vectors_May2021%22%2C%22type%22:%22image%22%2C%22blend%22:%22default%22%2C%22shader%22:%22void%20main%28%29%20%7B%20emitRGB%28vec3%28%28clamp%28getDataValue%280%29%2C%20-100.0%2C%20100.0%29+100.0%29/200.0%2C%20%28clamp%28getDataValue%281%29%2C%20-100.0%2C%20100.0%29+100.0%29/200.0%2C%20%28clamp%28getDataValue%282%29%2C%20-100.0%2C%20100.0%29+100.0%29/200.0%29%29%3B%20%7D%22%2C%22shaderControls%22:%7B%7D%2C%22name%22:%22vectors_May2021%22%7D%5D%2C%22navigation%22:%7B%22pose%22:%7B%22position%22:%7B%22voxelSize%22:%5B4.300000190734863%2C4.300000190734863%2C45%5D%2C%22voxelCoordinates%22:%5B42258.125%2C111597.5625%2C2220%5D%7D%7D%2C%22zoomFactor%22:5.466371588853563%7D%2C%22gpuMemoryLimit%22:4000000000%2C%22systemMemoryLimit%22:4000000000%2C%22concurrentDownloads%22:64%2C%22layout%22:%224panel%22%7D) shows the two layers output by the two CNNs: one predicts whether a given pixel is a postsynaptic site, and the other predicts vectors pointing from postsynaptic sites to presynaptic sites.

From the postsynaptic site probabilities, 83,917,332 discrete postsynaptic sites were extracted. The corresponding presynaptic sites were identified via the vector predictions at each postsynaptic site. The full set of ~84 million synaptic links can be found in the google cloud storage bucket [gs://zetta_lee_fly_vnc_001_alignment_temp/v4/fill_nearest_mip1/img/img_seethrough/synful_extraction/229bd2f77b2adf7c0e2c5b90ed605098/8.6_8.6_45/](https://console.cloud.google.com/storage/browser/zetta_lee_fly_vnc_001_alignment_temp/v4/fill_nearest_mip1/img/img_seethrough/synful_extraction/229bd2f77b2adf7c0e2c5b90ed605098/8.6_8.6_45)

## Filtering the predictions
A number of filters were applied to prune the ~84 million synaptic links down to a final set of ~50 million that constitute the final synapse table.

### Exclude synapses outside the segmentation
Any synaptic link where either the presynaptic or the postsynaptic site isn't associated with any reconstructed object (specifically, has no supervoxel at its position) was excluded. This filtered out 836,640 synapses (~1%), bringing the number of remaining synapses from 83,917,332 to 83,080,692.

### Exclude synapses with a score less than 12
The `synful` package provides a score for each predicted synaptic link. We found that thresholding the predictions by keeping only the ones with `sum_score > 12` (roughly meaning that the postsynaptic site had more than 12 voxels predicted to be a postsynaptic location) produced the maximal f-score when evaluating performance on ground truth synapse annotations. We applied this threshold, filtering out 26,566,091 synapses (~32%), bringing the number of remaining synapses from 83,080,692 to 56,514,601.

### Exclude autapses
Any synaptic link that connects a given supervoxel to itself was excluded. This removed 798,050 synapses (~1%), bringing the number of remaining synapses from 56,514,601 to 55,716,551.

### Exclude clear duplicates
Sometimes a single pair of supervoxels will be connected by two (or more) different synaptic links. This occurs due to limitations in the `synful` approach – in essentially all of these cases, the pair of supervoxels should only be connected a single time. In these cases, we removed any duplicates, leaving only the single link with the largest score to connect any given pair of supervoxels. This removed 5,724,380 synapses (~7%), bringing the number of remaining synapses from 55,716,551 to 49,992,171.

### Exclude likely duplicates
Similar to the section above, sometimes a single dendritic twig will be connected to the same presynaptic neuron by multiple links, but without having the same exact pair of supervoxels connected. We removed duplicate links that connect the same two segIDs if the links' presynaptic locations are within 150nm of one another. We examined the some of the cases identified by this approach and indeed all the instances we examined were duplicates that deserved to be removed. Here's a [neuroglancer link to 5 examples](https://neuromancer-seung-import.appspot.com/?json_url=https://global.daf-apis.com/nglstate/api/v1/5050799377874944). Thanks to Sven Dorkenwald for providing code for this step, which was also applied to the FAFB/FlyWire synapse table. This step removed 4,936,246 synapes (5.9%), bringing the number of remaining synapses from 49,992,171 to 45,055,925.

## Final Nov2022 synapse dataset
The final set of 45,055,925 synapses is available:
- as a csv file on google cloud storage at `gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/synapses_Nov2022/`. [Link to view files through browser](https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/alignmentV4/synapses/synapses_Nov2022).
- through a [CAVE](https://caveclient.readthedocs.io/en/latest/) table named `synapses_nov2022`. [Link to view table through browser](https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/synapses_nov2022).

The `score` column is the `sum_score` provided by `synful`, converted from float to int.


## Identify which region each (filtered) synapse is in

