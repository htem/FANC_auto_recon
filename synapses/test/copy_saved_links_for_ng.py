import sys
import json
from secrets import token_hex

import numpy as np

def line_anno(post, pre):
    anno = {
        'pointA': list(int(x) for x in post),
        'pointB': list(int(x) for x in pre),
        'type': 'line',
        'id': token_hex(40)
    }
    return anno


if sys.argv[1].endswith('.npy'):  # Works on .npy files saved from np.save
    output = np.load(sys.argv[1])
    flip_npy_from_zyx_to_xyz = True
    # Right now the .npy files I have are stored in zyx, so flip them to xyz
    if flip_npy_from_zyx_to_xyz:
        output[:,0:3] = output[:,2::-1]
        output[:,3:6] = output[:,5:2:-1]
else:  # Works on raw files saved from np.tofile
    output = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, 6)

annotations = [line_anno(output[i,0:3],output[i,3:6]) for i in range(output.shape[0])]
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
          "programmatically copy the output above to the clipboard")
