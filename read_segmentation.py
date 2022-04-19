import json
import os

ann_file = r'segmentacio_supervisely/210928_160015_k_r2_w_015_225_162/ds0/ann/210928_160015_k_r2_w_015_225_162.mp4'
ann_file = os.path.expanduser(ann_file) + '.json'

# Read annotations and store in a dict
with open(ann_file) as fh:
    data = fh.read()
annotations = json.loads(data)

print('finished')