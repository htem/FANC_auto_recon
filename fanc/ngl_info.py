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
      # As of Jan 2023, path is 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/em/rechunked'
      'path': info['aligned_volume']['image_source']}

seg = {'name': 'segmentation proofreading',
       # As of Aug 2022, path is 'graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod'
       'path': info['segmentation_source']}

syn = {'name': 'postsynapses',
       'path': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/postsynapses_May2021'}

# The soma_table's flat_segmentation_source is the un-_verified layer
#nuclei = {'name': 'nuclei_Mar2022',
#          'path': client.annotation.get_table_metadata(info['soma_table'])['flat_segmentation_source']
# so instead hardcode the _verified layer:
nuclei = {'name': 'nuclei (verified)',
          'path': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/nuclei/nuclei_seg_Mar2022_verified'}

view_options = dict(
    position=[48848, 114737, 2690],
    zoom_3d=6700,
    layout='xy-3d'
)
zoom_2d = 12


outlines_layer = {'type': 'segmentation', 'mesh': 'precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/volume_meshes/meshes', 'objectAlpha': 0.1, 'hideSegmentZero': False, 'ignoreSegmentInteractions': True, 'segmentColors': { '1': '#bfbfbf', '2': '#d343d6' }, 'segments': [ '1', '2' ], 'hiddenSegments': [ '104633', '104634', '104635', '104636', '104637', '104638', '104639', '104640', '104641', '104642', '104643', '104644', '104645', '104646', '104647', '104648', '104649', '104650', '104651', '104652', '104653' ], 'skeletonRendering': { 'mode2d': 'lines_and_points', 'mode3d': 'lines' }, 'name': 'region outlines'}



def final_json_tweaks(state):
    """
    Apply some final changes to the neuroglancer state that I didn't take the
    time to figure out how to do through nglui, by directly modifying the
    json/dict representation of the state.
    """
    for layer in state['layers']:
        if layer['name'] == seg['name']:
            layer['selectedAlpha'] = 0.4
        if layer['name'] == nuclei['name']:
            #layer['visible'] = False
            layer['ignoreSegmentInteractions'] = True
            layer['selectedAlpha'] = 0.8
        if layer['name'] == syn['name']:
            layer['visible'] = False
            layer['shader'] = 'void main() { emitRGBA(vec4(1, 0, 1, toNormalized(getDataValue()))); }'

    state['navigation']['zoomFactor'] = zoom_2d
    state.update({
        'gpuMemoryLimit': 4_000_000_000,
        'systemMemoryLimit': 4_000_000_000,
        'concurrentDownloads': 64,
        'jsonStateServer': 'https://global.daf-apis.com/nglstate/api/v1/post'
    })
