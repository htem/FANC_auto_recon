#!/usr/bin/env python3

import sys

import fanc


show_help = """\
    Render a neuron that has been reconstructed in FANC into an image stack aligned to a VNC template.
    The rendered neuron stack can be used for generation of a depth-colored MIP for use in mask searching.

    Usage:
      ./render_neuron_into_template_space.py seg_id [name_of_template_space]

    seg_id must be a segment ID from the FANC production segmentation.
    name_of_template_space must be the name of a template space. The options
      for this can be found in FANC_auto_recon/lm_em_comparisons/template_spaces.py
      If omitted, a default of 'JRC2018_VNC_UNISEX_461' will be used.

    Example usage:
      ./render_neuron_into_template_space.py 648518346494405175
    Output file will be named segid648518346494405175_in_JRC2018_VNC_UNISEX_461.nrrd

    WARNING: This script can take 10+ minutes to run on large neurons, as
      large neurons can have meshes with many millions of faces.
      A small neuron (~100k faces) that can be used for testing is 648518346516214999
"""


def main():
    if len(sys.argv) == 1:
        print(show_help)
        return

    seg_id = int(sys.argv[1])

    if len(sys.argv) >= 3:
        template_space_name = sys.argv[2]
    else:
        template_space_name = 'JRC2018_VNC_UNISEX_461'

    fanc.render_neuron_into_template_space(seg_id, template_space_name)


if __name__ == '__main__':
    main()
