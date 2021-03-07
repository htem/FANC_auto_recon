import sys
import json
from secrets import token_hex

import numpy as np

#sys.path.append('../ground_truth/')
#import synapse_cutout_rois
try:
    import synaptic_links
except ModuleNotFoundError:
    sys.path.append('../')
    import synaptic_links


def line_anno(pre, post):
    anno = {
        'pointA': list(int(x) for x in pre),
        'pointB': list(int(x) for x in post),
        'type': 'line',
        'id': token_hex(40)
    }
    return anno

links = synaptic_links.load(sys.argv[1])

annotations = [line_anno(links[i, 0:3], links[i, 3:6]) for i in range(links.shape[0])]
print(json.dumps(annotations, indent=2))
try:
    import pyperclip
    answer = input("Want to copy the output to the clipboard? (Only works if "
                   "you're running this script on a local machine, not on a "
                   "server.) [y/n] ")
    if answer.lower() == 'y':
        print('Copying')
        pyperclip.copy(json.dumps(annotations))
except:
    print("Install pyperclip (pip install pyperclip) for the option to"
          " programmatically copy the output above to the clipboard")
