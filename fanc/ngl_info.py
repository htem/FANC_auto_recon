#!/usr/bin/env python3
"""
Contains paths and settings relevant to building neuroglancer states
"""

from caveclient import CAVEclient


client = CAVEclient('fanc_production_mar2021')
info = client.info.get_datastack_info()

im = {'name': 'FANCv4',
      # As of Aug 2022, path is 'precomputed://gs://zetta_lee_fly_vnc_001_precomputed/fanc_v4_em'
      'path': info['aligned_volume']['image_source']}

seg = {'name': 'seg_Mar2021_proofreading',
       # As of Aug 2022, path is 'graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod'
       'path': info['segmentation_source']}

syn = {'name': 'synapses_May2021',
       'path': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/postsynapses_May2021'}

nuclei = {'name': 'nuclei_mar2022',
          'path': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/nuclei/nuclei_seg_Mar2022'}

view_options = dict(
    position=[48848, 114737, 2690],
    zoom_image=12,
    zoom_3d=6700
)


volume_meshes = {'type': 'segmentation', 'mesh': 'precomputed://gs://zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/brain_regions', 'objectAlpha': 0.1, 'hideSegmentZero': False, 'ignoreSegmentInteractions': True, 'segmentColors': { '1': '#bfbfbf', '2': '#d343d6' }, 'segments': [ '1', '2' ], 'hiddenSegments': [ '104633', '104634', '104635', '104636', '104637', '104638', '104639', '104640', '104641', '104642', '104643', '104644', '104645', '104646', '104647', '104648', '104649', '104650', '104651', '104652', '104653' ], 'skeletonRendering': { 'mode2d': 'lines_and_points', 'mode3d': 'lines' }, 'name': 'volume meshes'}

other_options = {
    'gpuMemoryLimit': 4_000_000_000,
    'systemMemoryLimit': 4_000_000_000,
    'concurrentDownloads': 64,
    'jsonStateServer': 'https://global.daf-apis.com/nglstate/api/v1/post'
}
