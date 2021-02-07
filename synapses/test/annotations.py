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


output = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, 6)

annotations = [line_anno(output[i,0:3],output[i,3:6]) for i in range(output.shape[0])]
print(json.dumps(annotations, indent=2))
