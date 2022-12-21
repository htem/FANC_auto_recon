#!/usr/bin/env python3
"""
Contains paths and settings relevant to building neuroglancer states
"""

from . import auth

client = auth.get_caveclient()
info = client.info.get_datastack_info()

ngl_app_url = info['viewer_site']
voxel_size = (info['viewer_resolution_x'],
              info['viewer_resolution_y'],
              info['viewer_resolution_z'])

im = {'name': 'FANCv4',
      # As of Aug 2022, path is 'precomputed://gs://zetta_lee_fly_vnc_001_precomputed/fanc_v4_em'
      'path': info['aligned_volume']['image_source']}

seg = {'name': 'seg_Mar2021_proofreading',
       # As of Aug 2022, path is 'graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod'
       'path': info['segmentation_source']}

syn = {'name': 'synapses_May2021',
       'path': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/postsynapses_May2021'}

nuclei = {'name': 'nuclei_Mar2022',
          'path': client.annotation.get_table_metadata(info['soma_table'])['flat_segmentation_source']}

view_options = dict(
    position=[48848, 114737, 2690],
    zoom_2d=12,
    zoom_3d=6700,
    layout='xy-3d'
)


volume_meshes = {'type': 'segmentation', 'mesh': 'precomputed://gs://zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/brain_regions', 'objectAlpha': 0.1, 'hideSegmentZero': False, 'ignoreSegmentInteractions': True, 'segmentColors': { '1': '#bfbfbf', '2': '#d343d6' }, 'segments': [ '1', '2' ], 'hiddenSegments': [ '104633', '104634', '104635', '104636', '104637', '104638', '104639', '104640', '104641', '104642', '104643', '104644', '104645', '104646', '104647', '104648', '104649', '104650', '104651', '104652', '104653' ], 'skeletonRendering': { 'mode2d': 'lines_and_points', 'mode3d': 'lines' }, 'name': 'volume meshes'}

other_options = {
    'gpuMemoryLimit': 4_000_000_000,
    'systemMemoryLimit': 4_000_000_000,
    'concurrentDownloads': 64,
    'jsonStateServer': 'https://global.daf-apis.com/nglstate/api/v1/post'
}
