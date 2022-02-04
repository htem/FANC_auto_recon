#/usr/bin/env python3
"""
Info on the stack dimensions and voxel sizes
of the different VNC standard templates
"""

template_info = {
    # Standard JRC2018 templates, available to download from
    # https://www.janelia.org/open-science/jrc-2018-brain-templates
    'JRC2018_VNC_FEMALE_4iso': {
        'stack dimensions': (660, 1342, 358),
        'voxel size': (0.4, 0.4, 0.4)
    },
    'JRC2018_VNC_MALE_4iso': {
        'stack dimensions': (659, 1342, 401),
        'voxel size': (0.4, 0.4, 0.4)
    },
    'JRC2018_VNC_UNISEX_4iso': {
        'stack dimensions': (660, 1290, 382),
        'voxel size': (0.4, 0.4, 0.4)
    },

    # The Gen1 VNC MCFO data are registered to versions of the standard
    # templates above that have been rescaled to a different voxel size.
    # The voxel size is 461.122 nm in x and y, so Janelia has chosen to
    # distribute these rescaled templates with filenames ending in '_461'.
    # These templates are available to download from
    # https://open.quiltdata.com/b/janelia-flylight-templates/tree/
    'JRC2018_VNC_FEMALE_461': {
        'stack dimensions': (573, 1164, 205),
        'voxel size': (0.461122, 0.461122, 0.7)
    },
    'JRC2018_VNC__MALE_461': {
        'stack dimensions': (572, 1164, 229),
        'voxel size': (0.461122, 0.461122, 0.7)
    },
    'JRC2018_VNC_UNISEX_461': {
        'stack dimensions': (573, 1119, 219),
        'voxel size': (0.461122, 0.461122, 0.7)
    }
}
